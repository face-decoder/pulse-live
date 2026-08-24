from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnBiLstmInferencer(BaseAnxietyInferencer):
    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_bi_lstm.cnn_bi_lstm import CNN_BiLSTM

        return CNN_BiLSTM(
            in_channels=BehavioralFeatures().n_channels,
            num_classes=2,
        )
