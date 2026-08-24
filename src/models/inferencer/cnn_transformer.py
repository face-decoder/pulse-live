from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnTransformerInferencer(BaseAnxietyInferencer):
    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_transformer.cnn_transformer import CNN_Transformer

        return CNN_Transformer(
            in_channels=BehavioralFeatures().n_channels,
            d_model=64,
            nhead=4,
            num_layers=2,
            num_classes=2,
        )
