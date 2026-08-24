from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnBiLstmMhaInferencer(BaseAnxietyInferencer):
    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_bi_lstm_mha.cnn_bi_lstm_mha import CNN_BiLSTM_MHA

        return CNN_BiLSTM_MHA(
            in_channels=BehavioralFeatures().n_channels,
            num_heads=4,
            num_classes=2,
        )
