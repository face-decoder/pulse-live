import torch.nn as nn


class SpatioTemporalCNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, dropout_p=0.4):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(16, num_classes),
        )

    def forward(self, x, mask=None):
        if x.ndim == 6:  # Handle ROI 6D (B, N_roi, C, T, H, W)
            B, N_roi, C, T, H, W = x.shape
            x = x.view(B, N_roi * C, T, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)
