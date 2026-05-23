import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Single-Head Attention use Pooling to flatten time dimension.
    Used as a replacement for x.mean(dim=1) to focus model on apex frame.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x, mask=None):
        # x shape: (Batch, Time, Hidden)
        attn_weights = self.attention(x)  # (Batch, Time, 1)

        if mask is not None:
            # mask shape: (Batch, Time) where True = padding
            # If using pooling, time length decreases, so mask needs to be adjusted.
            # Assumption: mask is already adjusted or not used (ignored with -inf)
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * x, dim=1)  # (Batch, Hidden)
        return context_vector, attn_weights
