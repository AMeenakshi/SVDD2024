import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model_attention import Model as TeacherModel
from compressed_sea_model import CompressedSEAModel

class ImprovedDistillationLoss(nn.Module):
    def __init__(self, alpha=0.6, tau=4.0, feature_weight=0.4):
        super().__init__()
        self.alpha = alpha  # Weight for distillation loss
        self.tau = tau      # Temperature for softmax
        self.feature_weight = feature_weight
        
    def forward(self, student_logits, teacher_logits, true_labels, hard_loss_fn,
                student_features=None, teacher_features=None):
        # Hard loss (student vs ground truth)
        hard_loss = hard_loss_fn(student_logits, true_labels)
        
        # Soft loss (student vs teacher predictions)
        soft_student = F.log_softmax(student_logits / self.tau, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.tau, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.tau ** 2)
        
        # Feature distillation loss
        feature_loss = 0
        if student_features is not None and teacher_features is not None:
            # SSL feature alignment
            if 'ssl' in student_features and 'ssl' in teacher_features:
                ssl_loss = F.mse_loss(student_features['ssl'], teacher_features['ssl'])
                feature_loss += ssl_loss
            
            # Attention map alignment
            if 'attention' in student_features and 'attention' in teacher_features:
                att_loss = F.kl_div(
                    F.log_softmax(student_features['attention'], dim=-1),
                    F.softmax(teacher_features['attention'], dim=-1),
                    reduction='batchmean'
                )
                feature_loss += att_loss
            
            # Graph feature alignment
            if 'graph_s' in student_features and 'graph_s' in teacher_features:
                graph_s_loss = F.mse_loss(student_features['graph_s'], teacher_features['graph_s'])
                graph_t_loss = F.mse_loss(student_features['graph_t'], teacher_features['graph_t'])
                feature_loss += (graph_s_loss + graph_t_loss)
        
        # Combined loss
        main_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        total_loss = main_loss + self.feature_weight * feature_loss
        
        return total_loss, {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'feature_loss': feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss,
            'total_loss': total_loss.item()
        }


class ImprovedKnowledgeDistillationModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        
        # Teacher model (your current model_attention.py)
        self.teacher = TeacherModel(args, device)
        self.teacher.eval()  # Always in eval mode
        
        # Student model (compressed but same architecture)
        self.student = CompressedSEAModel(args, device)
        
        # Feature alignment projectors (dimension matching)
        self.ssl_projector = nn.Linear(768, 1024)  # WavLM-Base to WavLM-Large
        self.attention_projector = nn.Linear(32, 64)  # Student attention to teacher
        self.graph_projector_s = nn.Linear(16, 32)  # Student graph dim to teacher
        self.graph_projector_t = nn.Linear(16, 32)
        
        # Distillation loss
        self.distill_criterion = ImprovedDistillationLoss(alpha=0.6, tau=4.0, feature_weight=0.4)
        
    def extract_teacher_features(self, x):
        """Extract intermediate features from teacher model"""
        with torch.no_grad():
            # SSL features
            teacher_ssl_feat = self.teacher.ssl_model(x)  # (bs, seq_len, 1024)
            
            # Process through initial layers
            x_processed = self.teacher.LL(teacher_ssl_feat)
            x_processed = x_processed.transpose(1, 2).unsqueeze(dim=1)
            x_processed = F.max_pool2d(x_processed, (3, 3))
            x_processed = self.teacher.first_bn(x_processed)
            x_processed = self.teacher.selu(x_processed)
            
            # Encoder features
            encoder_feat = self.teacher.encoder(x_processed)
            encoder_feat = self.teacher.first_bn1(encoder_feat)
            encoder_feat = self.teacher.selu(encoder_feat)
            
            # Attention maps
            attention_maps = self.teacher.attention(encoder_feat)
            
            # Spectral and temporal features
            w1 = F.softmax(attention_maps, dim=-1)
            m = torch.sum(encoder_feat * w1, dim=-1)
            e_S = m.transpose(1, 2) + self.teacher.pos_S
            
            w2 = F.softmax(attention_maps, dim=-2)
            m1 = torch.sum(encoder_feat * w2, dim=-2)
            e_T = m1.transpose(1, 2)
            
            # Graph features
            gat_S = self.teacher.GAT_layer_S(e_S)
            gat_T = self.teacher.GAT_layer_T(e_T)
            
            # Final prediction
            teacher_output = self.teacher(x)
            
            return {
                'ssl': teacher_ssl_feat,
                'attention': attention_maps,
                'graph_s': gat_S,
                'graph_t': gat_T,
                'output': teacher_output
            }
    
    def extract_student_features(self, x):
        """Extract intermediate features from student model"""
        # SSL features
        student_ssl_feat = self.student.ssl_model(x)  # (bs, seq_len, 768)
        
        # Process through initial layers
        x_processed = self.student.LL(student_ssl_feat)
        x_processed = x_processed.transpose(1, 2).unsqueeze(dim=1)
        x_processed = F.max_pool2d(x_processed, (3, 3))
        x_processed = self.student.first_bn(x_processed)
        x_processed = self.student.selu(x_processed)
        
        # Encoder features
        encoder_feat = self.student.encoder(x_processed)
        encoder_feat = self.student.first_bn1(encoder_feat)
        encoder_feat = self.student.selu(encoder_feat)
        
        # Attention maps
        attention_maps = self.student.attention(encoder_feat)
        
        # Spectral and temporal features
        w1 = F.softmax(attention_maps, dim=-1)
        m = torch.sum(encoder_feat * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.student.pos_S
        
        w2 = F.softmax(attention_maps, dim=-2)
        m1 = torch.sum(encoder_feat * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        
        # Graph features
        gat_S = self.student.GAT_layer_S(e_S)
        gat_T = self.student.GAT_layer_T(e_T)
        
        # Final prediction
        student_output = self.student(x)
        
        return {
            'ssl': student_ssl_feat,
            'attention': attention_maps,
            'graph_s': gat_S,
            'graph_t': gat_T,
            'output': student_output
        }
    
    def forward(self, x, return_features=False):
        if return_features:
            teacher_features = self.extract_teacher_features(x)
            student_features = self.extract_student_features(x)
            
            # Project student features to match teacher dimensions
            student_features_aligned = {
                'ssl': self.ssl_projector(student_features['ssl']),
                'attention': self.attention_projector(student_features['attention']),
                'graph_s': self.graph_projector_s(student_features['graph_s']),
                'graph_t': self.graph_projector_t(student_features['graph_t']),
                'output': student_features['output']
            }
            
            return {
                'student_features': student_features_aligned,
                'teacher_features': teacher_features,
                'student_output': student_features['output'],
                'teacher_output': teacher_features['output']
            }
        else:
            return self.student(x)
    
    def compute_distillation_loss(self, features, true_labels, hard_loss_fn):
        """Compute comprehensive distillation loss"""
        loss, loss_dict = self.distill_criterion(
            features['student_output'],
            features['teacher_output'],
            true_labels,
            hard_loss_fn,
            features['student_features'],
            features['teacher_features']
        )
        
        return loss, loss_dict


# Modified training function for improved distillation
def train_with_improved_distillation(model, train_loader, optimizer, criterion, device, epoch, writer):
    model.train()
    model.teacher.eval()  # Keep teacher in eval mode
    
    total_losses = {
        'total': 0, 'hard': 0, 'soft': 0, 'feature': 0
    }
    
    pos_samples, neg_samples = [], []
    
    for i, batch in enumerate(train_loader):
        x, labels, _ = batch
        x, labels = x.to(device), labels.to(device)
        
        # Forward pass with feature extraction
        features = model(x, return_features=True)
        
        # Compute distillation loss
        soft_labels = labels.float() * 0.9 + 0.05
        loss, loss_dict = model.compute_distillation_loss(
            features, soft_labels.unsqueeze(1), criterion
        )
        
        # Collect predictions from student for EER calculation
        student_pred = features['student_output']
        pos_samples.append(student_pred[labels == 1].detach().cpu().numpy())
        neg_samples.append(student_pred[labels == 0].detach().cpu().numpy())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Log detailed losses
        for key in total_losses:
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
        
        # Write to tensorboard
        if writer:
            step = epoch * len(train_loader) + i
            writer.add_scalar("Loss/total_train", loss_dict['total_loss'], step)
            writer.add_scalar("Loss/hard_train", loss_dict['hard_loss'], step)
            writer.add_scalar("Loss/soft_train", loss_dict['soft_loss'], step)
            writer.add_scalar("Loss/feature_train", loss_dict['feature_loss'], step)
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= len(train_loader)
    
    return total_losses, pos_samples, neg_samples