"""Inferencer for the CNN-BiLSTM-MHA architecture (series 0105, 0205, 0305, 0405)."""

from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnBiLstmMhaInferencer(BaseAnxietyInferencer):
    """Inferencer for :class:`CNN_BiLSTM_MHA`.

    Combination codes: **0105**, **0205**, **0305**, **0405**.

    Uses the standard behavioural-feature pipeline from
    :class:`BaseAnxietyInferencer` — ``BehavioralFeatures`` (47 channels),
    ``PadAndMask``, TTA forward pass.
    """

    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_bi_lstm_mha.cnn_bi_lstm_mha import CNN_BiLSTM_MHA

        return CNN_BiLSTM_MHA(
            in_channels=BehavioralFeatures().n_channels,
            num_heads=4,
            num_classes=2,
        )
