from __future__ import annotations

import torch
import torch.nn.functional as F

from .base_transform import BaseTransform
from .subject_sample import TransformOutput


class TemporalPool(BaseTransform):
    def __init__(self, target_len: int = 512):
        self.target_len = int(target_len)

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        x = inp.x
        if x.ndim != 2:
            raise ValueError(
                f"TemporalPool expects (T, C) input, got shape {tuple(x.shape)}"
            )
        x = x.permute(1, 0).unsqueeze(0)
        x = F.adaptive_avg_pool1d(x, self.target_len).squeeze(0)
        inp.x = x
        return inp


class PadAndMask(BaseTransform):
    def __init__(self, max_len: int = 512):
        self.max_len = int(max_len)

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        x = inp.x
        if x.ndim not in (2, 5):
            raise ValueError(
                f"PadAndMask expects 2D or 5D input, got shape {tuple(x.shape)}"
            )

        T_curr = x.shape[0]
        t = min(T_curr, self.max_len)

        padded_shape = (self.max_len,) + x.shape[1:]
        padded = torch.zeros(padded_shape, dtype=x.dtype)
        padded[:t] = x[:t]

        mask = torch.ones(self.max_len, dtype=torch.bool)
        mask[:t] = False

        if x.ndim == 2:
            inp.x = padded.permute(1, 0)
        elif x.ndim == 5:
            inp.x = padded.permute(1, 2, 0, 3, 4)

        inp.mask = mask
        return inp
