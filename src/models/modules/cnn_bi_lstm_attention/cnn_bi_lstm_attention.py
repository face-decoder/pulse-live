import torch.nn as nn

from ..attention_pooling import AttentionPooling
from ..cnn_1d_extractor import CNN1DExtractor


class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, in_channels=47, hidden_size=64, num_classes=2):
        super().__init__()
        self.cnn = CNN1DExtractor(in_channels, out_channels=hidden_size)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, mask=None):
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.bilstm(x)

        # Asumsi: CNN dengan maxpool(2) membagi panjang sequence menjadi dua.
        # Jika menggunakan mask asli, mask harus di-subsample juga.
        # Untuk kesederhanaan, kita abaikan mask di pooling layer jika CNN merubah dimensi waktu.
        context, attn_w = self.attention(x, mask=None)

        return self.classifier(context)
