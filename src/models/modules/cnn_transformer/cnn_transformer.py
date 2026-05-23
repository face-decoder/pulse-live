import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class CNN_Transformer(nn.Module):
    """
    Arsitektur Replikasi v12 murni (Transformer Encoder Only)
    """
    def __init__(self, in_channels=47, d_model=64, nhead=4, num_layers=2, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout_p, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        # x dari TemporalPool: (Batch, Channels, Time)
        x = x.permute(0, 2, 1) # Berubah menjadi (Batch, Time, Channels)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Masked Global Average Pooling (Bekerja sempurna dengan dummy mask)
        if mask is not None:
            valid_mask = ~mask
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(x).float()
            x_masked = x * valid_mask_expanded
            x_pooled = x_masked.sum(dim=1) / valid_mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            x_pooled = x.mean(dim=1)
            
        return self.classifier(x_pooled)