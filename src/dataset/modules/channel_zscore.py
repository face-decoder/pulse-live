from __future__ import annotations

from .base_transform import BaseTransform
from .subject_sample import TransformOutput


class ChannelZScore(BaseTransform):
    """
    Per-sample, per-channel z-score normalization.

    Input  x : (C, T)
    Output x : (C, T)  normalized
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        x = inp.x  # (C, T) or (N_roi, C, T, H, W)
        if x.ndim == 2:
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        elif x.ndim == 5:
            # Normalize per-channel, over spatial and temporal dimensions
            mu = x.mean(dim=(2, 3, 4), keepdim=True)
            sigma = x.std(dim=(2, 3, 4), keepdim=True, unbiased=False).clamp_min(self.eps)
        else:
            raise ValueError(f"ChannelZScore expects 2D or 5D, got {tuple(x.shape)}")
        inp.x = (x - mu) / sigma
        return inp
