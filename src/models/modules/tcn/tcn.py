import torch.nn as nn

from .temporal_block import TemporalBlock


class TCNModel(nn.Module):
    """
    Alternatif tanpa RNN yang sangat unggul untuk local temporal spikes.
    """

    def __init__(
        self, in_channels=47, num_channels=[64, 64, 64], kernel_size=3, num_classes=2
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_c = in_channels if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(
                TemporalBlock(in_c, out_c, kernel_size, dilation_size, padding)
            )

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, mask=None):
        # x shape: (Batch, Channels, Time)
        out = self.network(x)
        out = out.mean(dim=2)  # Global average pooling 1D
        return self.classifier(out)
