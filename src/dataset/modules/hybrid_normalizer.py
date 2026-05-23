from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


class HybridNormalizer:
    """
    Normalizer stateful untuk feature matrix (T, 47).

    Strategi per-channel:
      - Channel 0–19  (mean_dx, mean_dy, raw_mag, energy)  → Z-score
      - Channel 25–34 (accel, jerk)                        → Z-score
      - Channel 20–24 (dir_consistency)                    → MinMax [-1, 1]
      - Channel 35–46 (sync, symmetry)                     → MinMax [-1, 1]

    Alasan pembagian:
      - Fitur kinematik (dx, dy, mag, energy, accel, jerk) terdistribusi
        mendekati Gaussian → Z-score menghilangkan bias skala antar subjek.
      - Fitur bounded (dir_consist ∈ [0,1], sync ∈ [-1,1], sym ∈ [0,1])
        sudah dalam range tetap → MinMax lebih stabil daripada Z-score
        yang bisa membagi dengan std mendekati nol.

    Usage:
        normalizer = HybridNormalizer()
        train_samples = normalizer.fit_transform(train_samples)
        val_samples   = normalizer.transform(val_samples)

    Setiap sample adalah dict dengan key "signal": np.ndarray (T, C).
    Operasi in-place pada "signal"; key lain tidak tersentuh.
    """

    # Indices mengikuti layout BehavioralFeatureExtractor (C=47, N_roi=5)
    ZSCORE_CHANNELS = list(range(0, 20)) + list(range(25, 35))
    MINMAX_CHANNELS = list(range(20, 25)) + list(range(35, 47))

    def __init__(self) -> None:
        self.fitted = False
        self.mu_   : np.ndarray | None = None   # (1, C)
        self.std_  : np.ndarray | None = None   # (1, C)
        self.xmin_ : np.ndarray | None = None   # (1, C)
        self.xmax_ : np.ndarray | None = None   # (1, C)

    def fit(self, samples: List[dict]) -> "HybridNormalizer":
        """
        Hitung statistik dari training samples.

        Args:
            samples : list of dict dengan key "signal" → (T_i, C) float32

        Returns:
            self (untuk chaining)
        """
        if not samples:
            raise ValueError("samples kosong, tidak bisa fit.")

        all_features = np.concatenate(
            [s["signal"] for s in samples], axis=0
        )  # (sum(T_i), C)

        self.mu_   = all_features.mean(axis=0, keepdims=True)
        self.std_  = all_features.std(axis=0,  keepdims=True) + 1e-8
        self.xmin_ = all_features.min(axis=0,  keepdims=True)
        self.xmax_ = all_features.max(axis=0,  keepdims=True)
        self.fitted = True
        return self

    def transform(self, samples: List[dict]) -> List[dict]:
        """
        Terapkan normalisasi ke samples menggunakan statistik yang sudah fit.

        Operasi in-place pada samples[i]["signal"].
        """
        if not self.fitted:
            raise RuntimeError("Panggil .fit() atau .fit_transform() terlebih dahulu.")

        for s in samples:
            x = s["signal"].copy().astype(np.float32)

            # Z-score channels
            x[:, self.ZSCORE_CHANNELS] = (
                x[:, self.ZSCORE_CHANNELS] - self.mu_[:, self.ZSCORE_CHANNELS]
            ) / self.std_[:, self.ZSCORE_CHANNELS]

            # MinMax channels → [-1, 1]
            rng = self.xmax_[:, self.MINMAX_CHANNELS] - self.xmin_[:, self.MINMAX_CHANNELS]
            x[:, self.MINMAX_CHANNELS] = (
                2.0
                * (x[:, self.MINMAX_CHANNELS] - self.xmin_[:, self.MINMAX_CHANNELS])
                / (rng + 1e-8)
                - 1.0
            )

            s["signal"] = x

        return samples

    def fit_transform(self, samples: List[dict]) -> List[dict]:
        """Shorthand: fit lalu transform pada samples yang sama (training set)."""
        return self.fit(samples).transform(samples)

    def save(self, path: str | Path) -> None:
        """Simpan statistik normalizer ke file .npz."""
        if not self.fitted:
            raise RuntimeError("Normalizer belum di-fit.")
        np.savez(
            str(path),
            mu=self.mu_,
            std=self.std_,
            xmin=self.xmin_,
            xmax=self.xmax_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HybridNormalizer":
        """Load normalizer dari file .npz yang disimpan via .save()."""
        obj  = cls()
        data = np.load(str(path))
        obj.mu_    = data["mu"]
        obj.std_   = data["std"]
        obj.xmin_  = data["xmin"]
        obj.xmax_  = data["xmax"]
        obj.fitted = True
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        return (
            f"HybridNormalizer({status}, "
            f"zscore={len(self.ZSCORE_CHANNELS)}ch, "
            f"minmax={len(self.MINMAX_CHANNELS)}ch)"
        )