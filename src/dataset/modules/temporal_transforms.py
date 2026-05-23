from __future__ import annotations

import torch
import torch.nn.functional as F

from .base_transform import BaseTransform
from .subject_sample import TransformOutput


class TemporalPool(BaseTransform):
    """
    Pool sekuens ke fixed length dengan adaptive_avg_pool1d.

    Input  x : (T, C)
    Output x : (C, target_len)   → siap masuk transformer/CNN

    Note: Ini menghilangkan informasi panjang asli.
          Gunakan PadAndMask untuk preservasi panjang.
    """

    def __init__(self, target_len: int = 512):
        self.target_len = int(target_len)

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        x = inp.x  # (T, C)
        if x.ndim != 2:
            raise ValueError(
                f"TemporalPool expects (T, C) input, got shape {tuple(x.shape)}"
            )
        x = x.permute(1, 0).unsqueeze(0)  # (1, C, T)
        x = F.adaptive_avg_pool1d(x, self.target_len).squeeze(0)  # (C, target_len)
        inp.x = x
        return inp


class PadAndMask(BaseTransform):
    """
    Pad sekuens ke max_len dengan zeros; hasilkan boolean mask.

    Input  x    : (T, C)
    Output x    : (C, max_len)       → sudah di-transpose
           mask : (max_len,) bool    → True = posisi padding (diabaikan attention)

    Lebih baik dari TemporalPool karena preservasi panjang asli.
    Gunakan collate_pad_mask() untuk batching dengan mask ini.
    """

    def __init__(self, max_len: int = 512):
        self.max_len = int(max_len)

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        x = inp.x  # (T, C) or (T, N_roi, C, H, W)
        if x.ndim not in (2, 5):
            raise ValueError(f"PadAndMask expects 2D or 5D input, got shape {tuple(x.shape)}")

        T_curr = x.shape[0]
        t = min(T_curr, self.max_len)

        padded_shape = (self.max_len,) + x.shape[1:]
        padded = torch.zeros(padded_shape, dtype=x.dtype)
        padded[:t] = x[:t]

        mask = torch.ones(self.max_len, dtype=torch.bool)
        mask[:t] = False  # False = valid, True = ignore

        if x.ndim == 2:
            inp.x = padded.permute(1, 0)  # (C, max_len)
        elif x.ndim == 5:
            # (max_len, N_roi, C, H, W) -> (N_roi, C, max_len, H, W)
            inp.x = padded.permute(1, 2, 0, 3, 4)

        inp.mask = mask
        return inp
