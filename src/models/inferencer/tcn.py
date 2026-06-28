"""Inferencer for the TCN architecture (series 0106, 0206, 0306, 0406)."""

from __future__ import annotations

import torch.nn as nn

from .base import BaseAnxietyInferencer


class TcnInferencer(BaseAnxietyInferencer):
    """Inferencer for :class:`TCNModel`.

    Combination codes: **0106**, **0206**, **0306**, **0406**.

    Uses the standard behavioural-feature pipeline from
    :class:`BaseAnxietyInferencer` — ``BehavioralFeatures`` (47 channels),
    ``PadAndMask``, TTA forward pass.
    """

    def build_model(self) -> nn.Module:
        from src.dataset.modules.behavioral_features import BehavioralFeatures
        from src.models.modules.tcn.tcn import TCNModel

        return TCNModel(
            in_channels=BehavioralFeatures().n_channels,
            num_channels=[64, 64, 64],
            kernel_size=3,
            num_classes=2,
        )
