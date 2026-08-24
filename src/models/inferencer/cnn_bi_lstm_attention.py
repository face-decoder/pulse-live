from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnBiLstmAttentionInferencer(BaseAnxietyInferencer):
    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_bi_lstm_attention.cnn_bi_lstm_attention import (
            CNN_BiLSTM_Attention,
        )

        return CNN_BiLSTM_Attention(
            in_channels=BehavioralFeatures().n_channels,
            num_classes=2,
        )
