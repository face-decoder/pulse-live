"""Inferencer for the CNN-BiLSTM architecture (series 0103, 0203, 0303, 0403)."""

from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class CnnBiLstmInferencer(BaseAnxietyInferencer):
    """Inferencer for :class:`CNN_BiLSTM`.

    Combination codes: **0103**, **0203**, **0303**, **0403**.

    Uses the standard behavioural-feature pipeline from
    :class:`BaseAnxietyInferencer` — ``BehavioralFeatures`` (47 channels),
    ``PadAndMask``, TTA forward pass.
    """

    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.cnn_bi_lstm.cnn_bi_lstm import CNN_BiLSTM

        return CNN_BiLSTM(
            in_channels=BehavioralFeatures().n_channels,
            num_classes=2,
        )
