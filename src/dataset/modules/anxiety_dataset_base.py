from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..constants.index import LABEL_MAP
from .compose import Compose
from .subject_sample import SubjectSample, TransformOutput

logger = logging.getLogger(__name__)


class AnxietyDatasetBase(Dataset, ABC):
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "subject_id",
        "label",
        "is_valid",
        "npy_path",
    )
    LABEL_MAP = LABEL_MAP

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Compose | None = None,
        cache_dir: str | Path | None = None,
        force_rebuild: bool = False,
    ):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in metadata_df.columns]

        if missing:
            raise ValueError(f"Metadata has missing columns: {missing}")

        self.data = metadata_df[metadata_df["is_valid"].fillna(False)].copy()
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.force_rebuild = bool(force_rebuild)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._groups = self.__group_by_subject()
        self.subjects = list(self._groups.keys())
        self._mem_cache: dict[int, TransformOutput] = {}

    @abstractmethod
    def _load_flow(self, npy_path: str) -> np.ndarray: ...

    @abstractmethod
    def _detect_windows(self, flow: np.ndarray) -> list[tuple[int, int, int]]: ...

    def __group_by_subject(self) -> dict[str, pd.DataFrame]:
        return {sid: grp for sid, grp in self.data.groupby("subject_id", sort=True)}

    def __get_label(self, grp: pd.DataFrame) -> int:
        for raw_label in grp["label"].dropna().astype(str).str.strip().str.lower():
            if raw_label in self.LABEL_MAP:
                return self.LABEL_MAP[raw_label]
        raise ValueError(
            f"Tidak ada label valid untuk subject: "
            f"{grp['subject_id'].iloc[0]}. "
            f"Label tersedia: {list(grp['label'].unique())}"
        )

    def __build_subject_sample(self, subject_id: str) -> SubjectSample:
        grp = self._groups[subject_id]
        label = self.__get_label(grp)

        all_flows: list[np.ndarray] = []
        all_windows: list[tuple[int, int, int]] = []
        frame_offset: int = 0

        rows = (
            grp.sort_values("clip").iterrows()
            if "clip" in grp.columns
            else [(None, grp.iloc[i]) for i in range(len(grp))]
        )

        for _, row in rows:
            npy_path = str(row.get("npy_path", "")).strip()
            if not npy_path or not Path(npy_path).exists():
                logger.warning("[%s] File tidak ditemukan: %s", subject_id, npy_path)
                continue
            try:
                flow = self._load_flow(npy_path)
                windows = self._detect_windows(flow)

                for l, p, r in windows:
                    all_windows.append(
                        (
                            l + frame_offset,
                            p + frame_offset,
                            r + frame_offset,
                        )
                    )

                all_flows.append(flow)
                frame_offset += flow.shape[0]

            except Exception as exc:
                logger.warning("[%s] Gagal load %s: %s", subject_id, npy_path, exc)

        if len(all_flows) == 0:
            raise RuntimeError(f"Subjek {subject_id} tidak memiliki clip yang valid.")

        merged_flow = np.concatenate(all_flows, axis=0)

        meta: dict[str, Any] = {
            "subject_id": subject_id,
            "n_clips": len(all_flows),
            "n_windows": len(all_windows),
            "score": float(grp["score"].iloc[0]) if "score" in grp.columns else 0.0,
        }

        return SubjectSample(
            subject_id=subject_id,
            label=label,
            flow=merged_flow,
            windows=all_windows,
            meta=meta,
        )

    def __cache_key(self, subject_id: str) -> str:
        transform_repr = repr(self.transform) if self.transform else "none"
        return f"{subject_id}_{abs(hash(transform_repr)) % 10**8}"

    def __cache_path(self, subject_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.__cache_key(subject_id)}.pt"

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> TransformOutput:
        if index in self._mem_cache:
            return self._mem_cache[index]

        subject_id = self.subjects[index]
        cache_path = self.__cache_path(subject_id)

        if cache_path and cache_path.exists() and not self.force_rebuild:
            try:
                result = torch.load(cache_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                logger.warning("[%s] Cache load gagal, rebuild: %s", subject_id, exc)
                sample = self.__build_subject_sample(subject_id)
                if self.transform is not None:
                    result = self.transform(sample)
                else:
                    x = torch.from_numpy(sample.flow.astype(np.float32))
                    y = torch.tensor(sample.label, dtype=torch.long)
                    result = TransformOutput(x=x, y=y, meta=sample.meta)

                if cache_path:
                    torch.save(result, cache_path)
        else:
            sample = self.__build_subject_sample(subject_id)
            if self.transform is not None:
                result = self.transform(sample)
            else:
                x = torch.from_numpy(sample.flow.astype(np.float32))
                y = torch.tensor(sample.label, dtype=torch.long)
                result = TransformOutput(x=x, y=y, meta=sample.meta)

            if cache_path:
                torch.save(result, cache_path)

        self._mem_cache[index] = result
        return result

    def group_by_subject(self) -> dict[str, int]:
        return {sid: i for i, sid in enumerate(self.subjects)}

    def get_labels(self) -> np.ndarray:
        labels: list[int] = []
        for subject_id in self.subjects:
            grp = self._groups[subject_id]
            labels.append(self.__get_label(grp))
        return np.asarray(labels, dtype=np.int64)

    def get_class_counts(self) -> np.ndarray:
        labels = self.get_labels()
        return np.bincount(labels, minlength=2)

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        labels = self.get_labels()
        counts = np.bincount(labels, minlength=2)
        weights = (1.0 / np.maximum(counts, 1))[labels]
        sample_weights = torch.tensor(weights, dtype=torch.float32)
        return WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
