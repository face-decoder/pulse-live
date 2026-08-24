from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnLstmMlpInferencer(BaseAnxietyInferencer):
    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_lstm_mlp.cnn_lstm_mlp import CNN_LSTM_MLP

        return CNN_LSTM_MLP(
            in_channels=BehavioralFeatures().n_channels,
            num_classes=2,
        )
