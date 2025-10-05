import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model_attention import Model as TeacherModel

class CompressedSEAModel(nn.Module):
    """
    Student model that mimics teacher's SEA architecture but with reduced complexity
    This maintains architectural similarity while reducing parameters
    """
    def __init__(self, args, device):
        super(CompressedSEAModel, self).__init__()
        self.device = device
        
        # AASIST parameters (reduced from teacher)
        filts = [128, [1, 16], [16, 16], [16, 32], [32, 32]]  # Reduced from [1,32] to [1,16]
        gat_dims = [32, 16]  # Reduced from [64, 32] to [32, 16]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        # Compressed SSL model with similar architecture to teacher
        self.ssl_model = CompressedSSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 64)  # Reduced from 128 to 64

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=32)  # Reduced from 64 to 32
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # Compressed RawNet2 encoder (same structure, smaller dimensions)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        # Compressed attention (same concept, smaller dimensions)
        self.attention = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),  # Reduced from 64->128 to 32->64
            nn.SELU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=(1, 1)),  # Output 32 instead of 64
        )

        # Reduced position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 21, filts[-1][-1]))  # Reduced from 42 to 21
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Compressed Graph modules (same architecture, smaller dimensions)
        from Models.model_attention import GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool
        
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])

        # HS-GAL layer (compressed)
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 1)

    def forward(self, x):
        # Same forward pass as teacher but with compressed dimensions
        x = x.to(self.device)

        # Pre-trained WavLM model fine-tuning (compressed)
        x_ssl_feat = self.ssl_model(x)
        x = self.LL(x_ssl_feat)

        # Post-processing on front-end features
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)

        # SA for spectral feature
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # Graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # SA for temporal feature
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)

        # Graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # Learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # Inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # Inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


class CompressedSSLModel(nn.Module):
    """
    Compressed version of SSL model that maintains SEA architecture but reduces complexity
    """
    def __init__(self, device):
        super(CompressedSSLModel, self).__init__()
        from s3prl import hub
        
        self.model = getattr(hub, "wavlm_base")()  # Use WavLM-Base instead of Large
        self.device = device
        self.out_dim = 768  # WavLM-Base dimension (reduced from 1024)
        self.weight_hidd = nn.Parameter(torch.ones(30))

        # Compressed SE merge for wavlm-base of 12 layers (vs 25 for large)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_att_merge = nn.Sequential(
            nn.Linear(12, 4, bias=False),  # Reduced from 25->8 to 12->4
            nn.ReLU(inplace=True),
            nn.Linear(4, 12, bias=False),  # Reduced from 8->25 to 4->12
            nn.Sigmoid()
        )

        # Compressed Att merge for wavlm-base
        self.n_feat = 768  # Reduced from 1024
        self.n_layer = 12  # Reduced from 25
        self.W = nn.Parameter(torch.randn(self.n_feat, 1))
        self.W1 = nn.Parameter(torch.randn(self.n_layer, int(self.n_layer//2)))
        self.W2 = nn.Parameter(torch.randn(int(self.n_layer//2), self.n_layer))
        self.hidden = int(self.n_layer*self.n_feat/4)
        self.linear_proj = nn.Linear(self.n_layer*self.n_feat, self.n_feat)
        self.SWISH = nn.SiLU()

    def _SE_merge(self, x):
        """Compressed SE merge function"""
        feature = x['hidden_states']
        stacked_feature = torch.stack(feature, dim=1)
        b, c, _, _ = stacked_feature.size()
        y = self.avg_pool(stacked_feature).view(b, c)
        y = self.fc_att_merge(y).view(b, c, 1, 1)
        stacked_feature = stacked_feature * y.expand_as(stacked_feature)
        weighted_feature = torch.sum(stacked_feature, dim=1)
        return weighted_feature

    def _Att_merge(self, x):
        """Compressed attention merge function"""
        x = x['hidden_states']
        x = torch.stack(x, dim=1)
        x_input = x
        x = torch.mean(x, dim=2, keepdim=True)
        x = self.SWISH(torch.matmul(x, self.W))
        x = self.SWISH(torch.matmul(x.view(-1, self.n_layer), self.W1))
        x = torch.sigmoid((torch.matmul(x, self.W2)))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x, x_input)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2), -1)
        weighted_feature = self.linear_proj(x)
        return weighted_feature

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        if next(self.model.parameters()).device != input_data.device:
            self.model.to(input_data.device)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        emb = self.model(input_tmp)
        
        # Use both SE and Attention (like teacher) but compressed
        se_features = self._SE_merge(emb)
        att_features = self._Att_merge(emb)
        
        # Combine SE and Attention features
        combined_features = 0.7 * att_features + 0.3 * se_features
        return combined_features


# Copy Residual_block from model_attention.py
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out