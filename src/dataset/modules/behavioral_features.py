from __future__ import annotations

from itertools import combinations

import torch

from ..constants.index import ROI_ORDER_DEFAULT, SYMMETRY_PAIRS_DEFAULT
from .base_transform import BaseTransform
from .subject_sample import TransformOutput


class BehavioralFeatures(BaseTransform):
    def __init__(
        self,
        roi_order: list[str] = None,
        symmetry_pairs: list[tuple[int, int]] = None,
    ):
        self.roi_order = roi_order or ROI_ORDER_DEFAULT
        self.symmetry_pairs = symmetry_pairs or SYMMETRY_PAIRS_DEFAULT
        self.n_roi = len(self.roi_order)
        self.roi_pairs = list(combinations(range(self.n_roi), 2))
        self.n_sync = len(self.roi_pairs)
        self.n_sym = len(self.symmetry_pairs)
        self.n_channels = self.n_roi * 7 + self.n_sync + self.n_sym

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        flow = inp.x
        if not isinstance(flow, torch.Tensor):
            flow = torch.from_numpy(flow)

        if flow.ndim != 5:
            raise ValueError(
                f"BehavioralFeatures expects 5D flow (T,N,2,H,W), got {flow.ndim}D. "
                "Untuk FullFace gunakan BehavioralFeaturesFullFace."
            )

        device = flow.device
        if not flow.is_cuda and torch.cuda.is_available():
            flow = flow.cuda()

        features = self._extract(flow)

        inp.x = features.to(device)
        return inp

    def _extract(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[2] != 2:
            raise ValueError(f"Expects (T, N_roi, 2, H, W), got {flow.shape}")

        T, N_roi, _, H, W = flow.shape
        u = flow[:, :, 0, :, :]
        v = flow[:, :, 1, :, :]

        pixel_mag = torch.sqrt(u**2 + v**2)
        motion_energy = pixel_mag.mean(dim=(2, 3))

        angles = torch.atan2(v, u)

        mag_flat = pixel_mag.view(T, N_roi, -1)
        p25 = torch.quantile(mag_flat.float(), 0.25, dim=2, keepdim=True).to(
            mag_flat.dtype
        )
        mask = mag_flat > p25
        valid_counts = mask.sum(dim=2)

        ang_flat = angles.view(T, N_roi, -1)
        cos_ang = torch.cos(ang_flat) * mask
        sin_ang = torch.sin(ang_flat) * mask

        cos_sum = cos_ang.sum(dim=2)
        sin_sum = sin_ang.sum(dim=2)

        safe_counts = torch.where(
            valid_counts > 5, valid_counts, torch.ones_like(valid_counts)
        )
        cos_mean = cos_sum / safe_counts
        sin_mean = sin_sum / safe_counts

        R = torch.sqrt(cos_mean**2 + sin_mean**2)
        dir_consistency = torch.where(valid_counts > 5, R, torch.zeros_like(R))

        accel = torch.zeros_like(motion_energy)
        if T > 1:
            accel[1:] = torch.diff(motion_energy, dim=0)

        jerk = torch.zeros_like(motion_energy)
        if T > 2:
            jerk[2:] = torch.diff(motion_energy, n=2, dim=0)

        mean_dx = u.mean(dim=(2, 3))
        mean_dy = v.mean(dim=(2, 3))
        raw_mag = torch.sqrt(mean_dx**2 + mean_dy**2)

        mean_flow = torch.stack([mean_dx, mean_dy], dim=-1)
        sync = torch.zeros((T, self.n_sync), dtype=flow.dtype, device=flow.device)

        for idx, (i, j) in enumerate(self.roi_pairs):
            fi = mean_flow[:, i, :]
            fj = mean_flow[:, j, :]
            dot = (fi * fj).sum(dim=1)
            norm_i = torch.linalg.norm(fi, dim=1) + 1e-8
            norm_j = torch.linalg.norm(fj, dim=1) + 1e-8
            sync[:, idx] = dot / (norm_i * norm_j)

        sym = torch.zeros((T, self.n_sym), dtype=flow.dtype, device=flow.device)
        for idx, (li, ri) in enumerate(self.symmetry_pairs):
            el = motion_energy[:, li]
            er = motion_energy[:, ri]
            sym[:, idx] = torch.abs(el - er) / (el + er + 1e-8)

        features = torch.cat(
            [
                mean_dx,
                mean_dy,
                raw_mag,
                motion_energy,
                dir_consistency,
                accel,
                jerk,
                sync,
                sym,
            ],
            dim=1,
        ).float()

        return features

    def feature_names(self) -> list[str]:
        names = []
        for prefix in [
            "mean_dx",
            "mean_dy",
            "raw_mag",
            "energy",
            "dir_consist",
            "accel",
            "jerk",
        ]:
            names.extend([f"{prefix}_{r}" for r in self.roi_order])
        for i, j in self.roi_pairs:
            names.append(f"sync_{self.roi_order[i]}_{self.roi_order[j]}")
        for li, ri in self.symmetry_pairs:
            names.append(f"sym_{self.roi_order[li]}_{self.roi_order[ri]}")
        return names


class BehavioralFeaturesFullFace(BaseTransform):
    N_CHANNELS: int = 7

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        flow = inp.x
        if not isinstance(flow, torch.Tensor):
            flow = torch.from_numpy(flow)

        if flow.ndim != 4:
            raise ValueError(
                f"BehavioralFeaturesFullFace expects 4D flow (T,2,H,W), got {flow.ndim}D."
            )

        device = flow.device
        if not flow.is_cuda and torch.cuda.is_available():
            flow = flow.cuda()

        features = self._extract(flow)
        inp.x = features.to(device)
        return inp

    def _extract(self, flow: torch.Tensor) -> torch.Tensor:
        T = flow.shape[0]
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        mean_dx = u.mean(dim=(1, 2))
        mean_dy = v.mean(dim=(1, 2))
        magnitude = torch.sqrt(mean_dx**2 + mean_dy**2)

        pixel_mag = torch.sqrt(u**2 + v**2)
        energy = pixel_mag.mean(dim=(1, 2))

        angles = torch.atan2(v, u)
        cos_mean = torch.cos(angles).mean(dim=(1, 2))
        sin_mean = torch.sin(angles).mean(dim=(1, 2))
        dir_cons = torch.sqrt(cos_mean**2 + sin_mean**2)

        accel = torch.zeros(T, dtype=flow.dtype, device=flow.device)
        if T > 1:
            accel[1:] = torch.diff(energy, dim=0)

        jerk = torch.zeros(T, dtype=flow.dtype, device=flow.device)
        if T > 2:
            jerk[2:] = torch.diff(energy, n=2, dim=0)

        return torch.stack(
            [mean_dx, mean_dy, magnitude, energy, dir_cons, accel, jerk], dim=1
        ).float()
