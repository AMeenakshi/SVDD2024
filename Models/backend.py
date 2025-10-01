import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

class AASIST(nn.Module):
    def __init__(self, 
                 d_model=768, 
                 nhead=8, 
                 num_encoder_layers=6,
                 dim_feedforward=2048, 
                 dropout=0.1):
        super(AASIST, self).__init__()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Final classification layers
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, time, features)
        x = x.transpose(0, 1)  # (time, batch, features)
        
        # Apply transformer encoder
        out = self.transformer_encoder(x)
        
        # Global average pooling over time dimension
        out = torch.mean(out, dim=0)  # (batch, features)
        
        # Final classification
        out = self.output_layer(out)  # (batch, 1)
        
        return out