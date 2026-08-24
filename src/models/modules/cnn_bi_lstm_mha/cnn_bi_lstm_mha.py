import torch
import torch.nn as nn

from ..cnn_1d_extractor import CNN1DExtractor


class CNN_BiLSTM_MHA(nn.Module):
    def __init__(self, in_channels=47, hidden_size=64, num_heads=4, num_classes=2):
        super().__init__()
        self.cnn = CNN1DExtractor(in_channels, out_channels=hidden_size)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True,
        )

        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, mask=None):
        B = x.size(0)
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.bilstm(x)

        q = self.query.expand(B, -1, -1)
        context, attn_weights = self.mha(query=q, key=x, value=x)

        context = context.squeeze(1)
        return self.classifier(context)
