from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..constants.index import ROI_ORDER_DEFAULT

LABEL_MAP = {"anxiety_rendah": 0, "anxiety_tinggi": 1}
TARGET_NAMES = ["anxiety_rendah", "anxiety_tinggi"]


class HybridNormalizer:
    ZSCORE_CHANNELS = list(range(0, 20)) + list(range(25, 35))
    MINMAX_CHANNELS = list(range(20, 25)) + list(range(35, 47))

    def __init__(self):
        self.fitted = False
        self.mu_ = None
        self.std_ = None
        self.xmin_ = None
        self.xmax_ = None

    def fit(self, samples: list) -> "HybridNormalizer":
        all_f = np.concatenate([s["signal"] for s in samples], axis=0)
        self.mu_ = all_f.mean(axis=0, keepdims=True)
        self.std_ = all_f.std(axis=0, keepdims=True) + 1e-8
        self.xmin_ = all_f.min(axis=0, keepdims=True)
        self.xmax_ = all_f.max(axis=0, keepdims=True)
        self.fitted = True
        return self

    def transform(self, samples: list) -> list:
        assert self.fitted
        for s in samples:
            sig = s["signal"]
            if sig.ndim == 1:
                sig = sig[None, :]
            # zscore
            sig[:, self.ZSCORE_CHANNELS] = (sig[:, self.ZSCORE_CHANNELS] - self.mu_[:, self.ZSCORE_CHANNELS]) / self.std_[:, self.ZSCORE_CHANNELS]
            # minmax
            denom = (self.xmax_[:, self.MINMAX_CHANNELS] - self.xmin_[:, self.MINMAX_CHANNELS]) + 1e-8
            sig[:, self.MINMAX_CHANNELS] = (sig[:, self.MINMAX_CHANNELS] - self.xmin_[:, self.MINMAX_CHANNELS]) / denom
            s["signal"] = sig
        return samples

    def fit_transform(self, samples: list) -> list:
        return self.fit(samples).transform(samples)


def stratified_group_split(metadata_df, n_splits: int = 5, test_size: float = 0.2, random_state: int = 42):
    """Split a metadata dataframe by StratifiedGroupKFold on `subject_id` with labels from `label` column.

    Returns train_df, val_df, test_df (test_df may be empty if n_splits < 3).
    """
    if "subject_id" not in metadata_df.columns:
        raise ValueError("metadata_df must contain 'subject_id' column")
    if "label" not in metadata_df.columns:
        raise ValueError("metadata_df must contain 'label' column")

    subject_ids = metadata_df["subject_id"].values
    subject_labels = metadata_df["label"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, test_size=test_size, random_state=random_state)
    tr_idx, va_idx = next(sgkf.split(subject_ids, subject_labels, groups=subject_ids))
    train_subjects = set(subject_ids[tr_idx])
    val_subjects = set(subject_ids[va_idx])
    train_df = metadata_df[metadata_df["subject_id"].isin(train_subjects)].copy()
    val_df = metadata_df[metadata_df["subject_id"].isin(val_subjects)].copy()
    test_df = metadata_df[~metadata_df["subject_id"].isin(train_subjects | val_subjects)].copy()
    return train_df, val_df, test_df


def make_weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    weights = (1.0 / np.maximum(counts, 1))[labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32), num_samples=len(labels), replacement=True)


def get_loaders(train_df, val_df, test_df, dataset_prototype, batch_size: int = 8, max_seq_len: int = 512):
    """Create DataLoaders given metadata dataframes and a prototype dataset instance.

    The function will instantiate new dataset objects of the same class as `dataset_prototype`
    using the subset metadata DataFrames and any matching constructor kwargs found on the prototype.
    """
    proto = dataset_prototype
    cls = proto.__class__

    def _make(df):
        # pick commonly used attributes if present on prototype
        kw = {}
        for a in ["detector", "extractor", "phase_mode", "selected_rois", "transform", "cache_dir", "force_rebuild", "roi_order"]:
            if hasattr(proto, a):
                kw[a] = getattr(proto, a)
        return cls(metadata_df=df, **kw)

    train_dataset = _make(train_df)
    val_dataset = _make(val_df)
    test_dataset = _make(test_df)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
