from __future__ import annotations

from pathlib import Path

import numpy as np


class HybridNormalizer:
    ZSCORE_CHANNELS = list(range(20)) + list(range(25, 35))
    MINMAX_CHANNELS = list(range(20, 25)) + list(range(35, 47))

    def __init__(self) -> None:
        self.fitted = False
        self.mu_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.xmin_: np.ndarray | None = None
        self.xmax_: np.ndarray | None = None

    def fit(self, samples: list[dict]) -> HybridNormalizer:
        if not samples:
            raise ValueError("samples kosong, tidak bisa fit.")

        all_features = np.concatenate([s["signal"] for s in samples], axis=0)

        self.mu_ = all_features.mean(axis=0, keepdims=True)
        self.std_ = all_features.std(axis=0, keepdims=True) + 1e-8
        self.xmin_ = all_features.min(axis=0, keepdims=True)
        self.xmax_ = all_features.max(axis=0, keepdims=True)
        self.fitted = True
        return self

    def transform(self, samples: list[dict]) -> list[dict]:
        if not self.fitted:
            raise RuntimeError("Panggil .fit() atau .fit_transform() terlebih dahulu.")

        for s in samples:
            x = s["signal"].copy().astype(np.float32)

            x[:, self.ZSCORE_CHANNELS] = (
                x[:, self.ZSCORE_CHANNELS] - self.mu_[:, self.ZSCORE_CHANNELS]
            ) / self.std_[:, self.ZSCORE_CHANNELS]

            rng = (
                self.xmax_[:, self.MINMAX_CHANNELS]
                - self.xmin_[:, self.MINMAX_CHANNELS]
            )
            x[:, self.MINMAX_CHANNELS] = (
                2.0
                * (x[:, self.MINMAX_CHANNELS] - self.xmin_[:, self.MINMAX_CHANNELS])
                / (rng + 1e-8)
                - 1.0
            )

            s["signal"] = x

        return samples

    def fit_transform(self, samples: list[dict]) -> list[dict]:
        return self.fit(samples).transform(samples)

    def save(self, path: str | Path) -> None:
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
    def load(cls, path: str | Path) -> HybridNormalizer:
        obj = cls()
        data = np.load(str(path))
        obj.mu_ = data["mu"]
        obj.std_ = data["std"]
        obj.xmin_ = data["xmin"]
        obj.xmax_ = data["xmax"]
        obj.fitted = True
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        return (
            f"HybridNormalizer({status}, "
            f"zscore={len(self.ZSCORE_CHANNELS)}ch, "
            f"minmax={len(self.MINMAX_CHANNELS)}ch)"
        )
