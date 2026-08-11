"""Inferencer for the CNN-BiLSTM-Attention architecture (series 0104, 0204, 0304, 0404)."""

from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnBiLstmAttentionInferencer(BaseAnxietyInferencer):
    """Inferencer for :class:`CNN_BiLSTM_Attention`.

    Combination codes: **0104**, **0204**, **0304**, **0404**.

    Uses the standard behavioural-feature pipeline from
    :class:`BaseAnxietyInferencer` — ``BehavioralFeatures`` (47 channels),
    ``PadAndMask``, TTA forward pass.
    """

    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_bi_lstm_attention.cnn_bi_lstm_attention import (
            CNN_BiLSTM_Attention,
        )

        return CNN_BiLSTM_Attention(
            in_channels=BehavioralFeatures().n_channels,
            num_classes=2,
        )
