import torch.nn as nn
import torch.nn.functional as F


class CNN1DExtractor(nn.Module):
    """
    Temporal feature extractor.
    Change input shape from (Batch, In_Channels, Time) to (Batch, Out_Channels, Time/Pool).
    """

    def __init__(self, in_channels, out_channels=64, pool_size=2, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels // 2)
        self.drop1 = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            out_channels // 2, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.pool = (
            nn.MaxPool1d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(x)
        return x
