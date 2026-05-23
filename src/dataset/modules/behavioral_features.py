from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import circvar

from ..constants.index import ROI_ORDER_DEFAULT, SYMMETRY_PAIRS_DEFAULT
from .base_transform import BaseTransform
from .subject_sample import TransformOutput


class BehavioralFeatures(BaseTransform):
    """
    Ekstrak 47-channel behavioral features dari raw flow window.

    Input  x : (T, N_roi, 2, H, W)   [ROI mode]
    Output x : (T, C_behavioral)

    Channels (5 ROI → 47 total):
      mean_dx        : N_roi
      mean_dy        : N_roi
      raw_magnitude  : N_roi
      motion_energy  : N_roi
      dir_consistency: N_roi
      acceleration   : N_roi
      jerk           : N_roi
      sync (pairwise): C(N_roi, 2) = 10
      symmetry       : len(symmetry_pairs) = 2
    ──────────────────
    Total             : 5*7 + 10 + 2 = 47

    Notes:
        - Hanya berlaku untuk ROI flow (5D).
        - Untuk FullFace flow (4D), gunakan BehavioralFeaturesFullFace.
    """

    def __init__(
        self,
        roi_order: List[str] = None,
        symmetry_pairs: List[Tuple[int, int]] = None,
    ):
        self.roi_order = roi_order or ROI_ORDER_DEFAULT
        self.symmetry_pairs = symmetry_pairs or SYMMETRY_PAIRS_DEFAULT
        self.n_roi = len(self.roi_order)
        self.roi_pairs = list(combinations(range(self.n_roi), 2))
        self.n_sync = len(self.roi_pairs)
        self.n_sym = len(self.symmetry_pairs)
        self.n_channels = self.n_roi * 7 + self.n_sync + self.n_sym

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        # inp.x: (T, N_roi, 2, H, W) — raw flow tensor
        flow_np = inp.x.numpy()

        if flow_np.ndim != 5:
            raise ValueError(
                f"BehavioralFeatures expects 5D flow (T,N,2,H,W), got {flow_np.ndim}D. "
                "Untuk FullFace gunakan BehavioralFeaturesFullFace."
            )

        features = self._extract(flow_np)  # (T, C)
        inp.x = torch.from_numpy(features)
        return inp

    def _extract(self, flow: np.ndarray) -> np.ndarray:
        # Use the v12-style extractor logic (magnitude-masked circvar)
        if flow.ndim != 5 or flow.shape[2] != 2:
            raise ValueError(f"Expects (T, N_roi, 2, H, W), got {flow.shape}")

        T, N_roi, _, H, W = flow.shape
        u = flow[:, :, 0, :, :]
        v = flow[:, :, 1, :, :]

        pixel_mag = np.sqrt(u ** 2 + v ** 2)
        motion_energy = pixel_mag.mean(axis=(2, 3))

        angles = np.arctan2(v, u)
        dir_consistency = np.zeros((T, N_roi), dtype=np.float32)
        for t in range(T):
            for r in range(N_roi):
                ang_flat = angles[t, r].ravel()
                mag_flat = pixel_mag[t, r].ravel()
                mask = mag_flat > np.percentile(mag_flat, 25)
                if mask.sum() > 5:
                    cv = circvar(ang_flat[mask], high=np.pi, low=-np.pi)
                    dir_consistency[t, r] = 1.0 - float(cv)

        accel = np.zeros_like(motion_energy)
        if T > 1:
            accel[1:] = np.diff(motion_energy, axis=0)

        jerk = np.zeros_like(motion_energy)
        if T > 2:
            jerk[2:] = np.diff(motion_energy, n=2, axis=0)

        mean_dx = u.mean(axis=(2, 3))
        mean_dy = v.mean(axis=(2, 3))
        raw_mag = np.sqrt(mean_dx ** 2 + mean_dy ** 2)

        mean_flow = np.stack([mean_dx, mean_dy], axis=-1)
        sync = np.zeros((T, self.n_sync), dtype=np.float32)
        for idx, (i, j) in enumerate(self.roi_pairs):
            fi = mean_flow[:, i, :]
            fj = mean_flow[:, j, :]
            dot = (fi * fj).sum(axis=1)
            norm_i = np.linalg.norm(fi, axis=1) + 1e-8
            norm_j = np.linalg.norm(fj, axis=1) + 1e-8
            sync[:, idx] = dot / (norm_i * norm_j)

        sym = np.zeros((T, self.n_sym), dtype=np.float32)
        for idx, (li, ri) in enumerate(self.symmetry_pairs):
            el = motion_energy[:, li]
            er = motion_energy[:, ri]
            sym[:, idx] = np.abs(el - er) / (el + er + 1e-8)

        features = np.concatenate([
            mean_dx,
            mean_dy,
            raw_mag,
            motion_energy,
            dir_consistency,
            accel,
            jerk,
            sync,
            sym,
        ], axis=1).astype(np.float32)

        return features

    def feature_names(self) -> List[str]:
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
    """
    Behavioral features untuk FullFace flow (4D).

    Input  x : (T, 2, H, W)
    Output x : (T, C_ff)

    Channels:
      mean_dx     : 1
      mean_dy     : 1
      magnitude   : 1
      energy      : 1
      dir_consistency : 1
      acceleration: 1
      jerk        : 1
    ─────────────
    Total         : 7
    """

    N_CHANNELS: int = 7

    def __call__(self, inp: TransformOutput) -> TransformOutput:
        flow_np = inp.x.numpy()

        if flow_np.ndim != 4:
            raise ValueError(
                f"BehavioralFeaturesFullFace expects 4D flow (T,2,H,W), got {flow_np.ndim}D."
            )

        features = self._extract(flow_np)
        inp.x = torch.from_numpy(features)
        return inp

    def _extract(self, flow: np.ndarray) -> np.ndarray:
        T = flow.shape[0]
        u = flow[:, 0, :, :]  # (T, H, W) dx
        v = flow[:, 1, :, :]  # (T, H, W) dy

        mean_dx = u.mean(axis=(1, 2), keepdims=False)  # (T,)
        mean_dy = v.mean(axis=(1, 2), keepdims=False)  # (T,)
        magnitude = np.sqrt(mean_dx**2 + mean_dy**2)  # (T,)

        pixel_mag = np.sqrt(u**2 + v**2)
        energy = pixel_mag.mean(axis=(1, 2))  # (T,)

        # Vectorized circular variance
        angles = np.arctan2(v, u)
        cos_mean = np.cos(angles).mean(axis=(1, 2))
        sin_mean = np.sin(angles).mean(axis=(1, 2))
        dir_cons = np.sqrt(cos_mean**2 + sin_mean**2)  # (T,)

        accel = np.zeros(T, dtype=np.float32)
        if T > 1:
            accel[1:] = np.diff(energy)

        jerk = np.zeros(T, dtype=np.float32)
        if T > 2:
            jerk[2:] = np.diff(energy, n=2)

        return np.stack(
            [mean_dx, mean_dy, magnitude, energy, dir_cons, accel, jerk], axis=1
        ).astype(np.float32)  # (T, 7)
