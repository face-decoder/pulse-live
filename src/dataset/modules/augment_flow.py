from __future__ import annotations

from typing import Tuple

import torch

from .base_transform import BaseTransform
from .subject_sample import TransformOutput


class AugmentFlow(BaseTransform):
    """
    Augmentasi untuk tensor (C, T) — hanya aktif saat training.

    Pipeline augmentasi (berurutan):
      1. Magnitude scaling  : x *= Uniform(scale_lo, scale_hi)
      2. Temporal jitter    : roll ±jitter_frames, zero-fill wrapped region
      3. Temporal masking   : zero out a short time segment
      4. Channel dropout    : zero out random channels dengan prob dropout_p
      5. Gaussian noise     : x += N(0, noise_std)

    Args:
        training    : jika False, transform ini menjadi no-op
        scale_range : (lo, hi) untuk magnitude scaling
        jitter_frames: max shift frames
        temporal_mask_prob : probabilitas temporal masking
        temporal_mask_ratio: (lo, hi) panjang mask relatif terhadap T
        dropout_p   : prob zero out tiap channel (independent)
        noise_std   : std Gaussian noise
    """

    def __init__(
        self,
        training: bool = True,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        jitter_frames: int = 2,
        # temporal_mask_prob: float = 0.2,
        # temporal_mask_ratio: Tuple[float, float] = (0.03, 0.08),
        dropout_p: float = 0.1,
        noise_std: float = 0.01,
    ):
        self.training = bool(training)
        self.scale_lo = float(scale_range[0])
        self.scale_hi = float(scale_range[1])
        self.jitter_frames = int(jitter_frames)
        # self.temporal_mask_prob = float(temporal_mask_prob)
        # self.temporal_mask_lo = float(temporal_mask_ratio[0])
        # self.temporal_mask_hi = float(temporal_mask_ratio[1])
        self.dropout_p = float(dropout_p)
        self.noise_std = float(noise_std)

    def train(self) -> "AugmentFlow":
        self.training = True
        return self

    def eval(self) -> "AugmentFlow":
        self.training = False
        return self

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        if not self.training:
            return inp

        x = inp.x  # (C, T) or (N_roi, C, T, H, W)

        # 1. Magnitude scaling
        if self.scale_lo < self.scale_hi:
            scale = self.scale_lo + torch.rand(1).item() * (
                self.scale_hi - self.scale_lo
            )
            x = x * scale

        # 2. Temporal jitter
        if self.jitter_frames > 0:
            shift = int(
                torch.randint(-self.jitter_frames, self.jitter_frames + 1, (1,)).item()
            )
            if shift != 0:
                dim_T = 1 if x.ndim == 2 else 2
                x = torch.roll(x, shifts=shift, dims=dim_T)
                if shift > 0:
                    if x.ndim == 2:
                        x[:, :shift] = 0.0
                    else:
                        x[:, :, :shift, :, :] = 0.0
                else:
                    if x.ndim == 2:
                        x[:, shift:] = 0.0
                    else:
                        x[:, :, shift:, :, :] = 0.0

        # 3. Temporal masking (short segment)
        # if self.temporal_mask_prob > 0 and self.temporal_mask_lo < self.temporal_mask_hi:
        #     if torch.rand(1).item() < self.temporal_mask_prob:
        #         dim_T = 1 if x.ndim == 2 else 2
        #         T = x.shape[dim_T]
        #         if T > 1:
        #             frac = self.temporal_mask_lo + torch.rand(1).item() * (
        #                 self.temporal_mask_hi - self.temporal_mask_lo
        #             )
        #             mask_len = max(1, int(round(T * frac)))
        #             start = int(torch.randint(0, max(1, T - mask_len + 1), (1,)).item())
        #             if x.ndim == 2:
        #                 x[:, start : start + mask_len] = 0.0
        #             else:
        #                 x[:, :, start : start + mask_len, :, :] = 0.0

        # 4. Channel dropout
        if self.dropout_p > 0:
            if x.ndim == 2:
                mask = (torch.rand(x.shape[0], 1) > self.dropout_p).float()
            elif x.ndim == 5:
                # Dropout independent for each ROI and Channel
                mask = (torch.rand(x.shape[0], x.shape[1], 1, 1, 1) > self.dropout_p).float()
            x = x * mask

        # 5. Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        inp.x = x
        return inp
