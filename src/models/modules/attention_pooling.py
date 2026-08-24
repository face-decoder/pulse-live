import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x, mask=None):
        attn_weights = self.attention(x)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * x, dim=1)
        return context_vector, attn_weights
