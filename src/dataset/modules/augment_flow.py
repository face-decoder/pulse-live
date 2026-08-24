from __future__ import annotations

import torch

from .base_transform import BaseTransform
from .subject_sample import TransformOutput


class AugmentFlow(BaseTransform):
    def __init__(
        self,
        training: bool = True,
        scale_range: tuple[float, float] = (0.85, 1.15),
        jitter_frames: int = 2,
        dropout_p: float = 0.1,
        noise_std: float = 0.01,
    ):
        self.training = bool(training)
        self.scale_lo = float(scale_range[0])
        self.scale_hi = float(scale_range[1])
        self.jitter_frames = int(jitter_frames)
        self.dropout_p = float(dropout_p)
        self.noise_std = float(noise_std)

    def train(self) -> AugmentFlow:
        self.training = True
        return self

    def eval(self) -> AugmentFlow:
        self.training = False
        return self

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        if not self.training:
            return inp

        x = inp.x

        if self.scale_lo < self.scale_hi:
            scale = self.scale_lo + torch.rand(1).item() * (
                self.scale_hi - self.scale_lo
            )
            x = x * scale

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

        if self.dropout_p > 0:
            if x.ndim == 2:
                mask = (torch.rand(x.shape[0], 1) > self.dropout_p).float()
            elif x.ndim == 5:
                mask = (
                    torch.rand(x.shape[0], x.shape[1], 1, 1, 1) > self.dropout_p
                ).float()
            x = x * mask

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        inp.x = x
        return inp
