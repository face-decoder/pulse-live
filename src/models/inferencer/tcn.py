from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class TcnInferencer(BaseAnxietyInferencer):
    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.tcn.tcn import TCNModel

        return TCNModel(
            in_channels=BehavioralFeatures().n_channels,
            num_channels=[64, 64, 64],
            kernel_size=3,
            num_classes=2,
        )
