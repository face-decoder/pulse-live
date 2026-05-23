import torch.nn as nn

from ..cnn_1d_extractor import CNN1DExtractor


class CNN_BiLSTM(nn.Module):
    def __init__(self, in_channels=47, hidden_size=64, num_classes=2):
        super().__init__()
        self.cnn = CNN1DExtractor(in_channels, out_channels=hidden_size)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, mask=None):
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.bilstm(x)  # (Batch, Time, Hidden)

        # Mean pooling sepanjang dimensi waktu
        x_pooled = x.mean(dim=1)
        return self.classifier(x_pooled)
