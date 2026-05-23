import torch.nn as nn

from ..cnn_1d_extractor import CNN1DExtractor


class CNN_LSTM_MLP(nn.Module):
    def __init__(self, in_channels=47, hidden_size=64, num_classes=2):
        super().__init__()
        self.cnn = CNN1DExtractor(in_channels, out_channels=hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, mask=None):
        # x shape: (Batch, Channels, Time)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Menjadi (Batch, Time, Channels) untuk LSTM

        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # Ambil state terakhir

        return self.classifier(last_hidden)
