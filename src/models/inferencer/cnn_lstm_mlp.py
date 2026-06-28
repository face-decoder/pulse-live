"""Inferencer for the CNN-LSTM-MLP architecture (series 0102, 0202, 0302, 0402)."""

from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnLstmMlpInferencer(BaseAnxietyInferencer):
    """Inferencer for :class:`CNN_LSTM_MLP`.

    Combination codes: **0102**, **0202**, **0302**, **0402**.

    Uses the standard behavioural-feature pipeline from
    :class:`BaseAnxietyInferencer` — ``BehavioralFeatures`` (47 channels),
    ``PadAndMask``, TTA forward pass.
    """

    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_lstm_mlp.cnn_lstm_mlp import CNN_LSTM_MLP

        return CNN_LSTM_MLP(
            in_channels=BehavioralFeatures().n_channels,
            num_classes=2,
        )
