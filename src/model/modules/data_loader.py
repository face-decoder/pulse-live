from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from .feature_extractor import FeatureExtractor
from src.apex.modules.v2 import ApexPhase

logger = logging.getLogger(__name__)

_VALID_MODES = ("train", "eval")
_VALID_STAGES = ("before", "after", "all")


class MicroExpressionDataset(Dataset):
    """Dataset for micro-expression anxiety training and evaluation.

    Attributes:
        K_APEX: Number of top apex frames used by FeatureExtractor.
        mode: Current dataset mode (``"train"`` or ``"eval"``).
        annotations: Filtered DataFrame of valid annotation rows.
        skipped_clips: DataFrame of rows whose ``.npy`` files were
            not found during initialisation.

    Example::

        ds = MicroExpressionDataset("train_annotations_v3.csv", stage="all")
        folds = MicroExpressionDataset.build_group_kfold(
            ds.annotations, n_splits=5,
        )
        for fold_info in folds:
            train_sub = Subset(ds, fold_info["train_idx"])
    """

    K_APEX: int = FeatureExtractor.K_APEX  # 3

    # CSV string  →  numeric label
    _CSV_TO_LABEL: dict[str, int] = {
        "high": 1,
        "low": 0,
    }

    def __init__(
        self,
        annotations_file: str,
        stage: str = "all",
        mode: str = "train",
        valid_only: bool = True,
    ) -> None:
        """Initialise the dataset.

        Args:
            annotations_file: Path to ``train_annotations_v3.csv``.
            stage: Filter rows by the ``stage`` column.
                One of ``"before"``, ``"after"``, or ``"all"``.
            mode: ``"train"`` enables augmentation via
                :meth:`FeatureExtractor.augment`;
                ``"eval"`` disables it.
            valid_only: If ``True``, keep only rows where
                ``is_valid != False``.  The ``is_valid`` column is
                optional — all rows are considered valid if absent.

        Raises:
            ValueError: If *mode* or *stage* is not a recognised value.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"'mode' must be one of {_VALID_MODES}, got '{mode}'"
            )
        if stage not in _VALID_STAGES:
            raise ValueError(
                f"'stage' must be one of {_VALID_STAGES}, got '{stage}'"
            )

        self.mode = mode

        df = pd.read_csv(annotations_file)

        # Normalise mandatory columns
        df["stage"] = df["stage"].astype(str).str.strip().str.lower()
        df["anxiety_level"] = df["anxiety_level"].astype(str).str.strip().str.lower()
        df["subject_id"] = df["subject_id"].astype(str).str.strip()

        # Filter by stage
        if stage != "all":
            df = df[df["stage"] == stage]

        # Filter by is_valid
        if valid_only and "is_valid" in df.columns:
            df = df[df["is_valid"].astype(str).str.lower() != "false"]

        self.annotations = df.reset_index(drop=True)

        # Scan and skip rows whose .npy files are missing.
        # Done once in __init__ so __getitem__ never needs to check.
        self.annotations, self.skipped_clips = self._scan_missing(self.annotations)

        # Internal components
        self._extractor = FeatureExtractor()
        self._apex_detector = ApexPhase()
        self._apex_cache: dict[str, tuple[np.ndarray, dict]] = {}

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return ``(features, label, subject_id)`` for one clip.

        Args:
            idx: Row index into :attr:`annotations`.

        Returns:
            A 3-tuple of ``(feat, label, subject)``:

            - ``feat``    — ``torch.Tensor`` of shape ``(78,)``
            - ``label``   — ``torch.Tensor`` scalar (``torch.long``)
            - ``subject`` — ``str`` subject_id
        """
        row = self.annotations.iloc[idx]

        label = self._parse_label(row["anxiety_level"])
        subject = str(row["subject_id"])
        npy_path = str(row["npy_path"])

        # File is guaranteed to exist (filtered in __init__)
        loaded = np.load(npy_path, allow_pickle=True).item()
        roi_frames = loaded["frames"]
        magnitudes = np.asarray(loaded["magnitudes"], dtype=np.float32)

        # Apex detection with caching (avoids re-computation each epoch)
        if npy_path not in self._apex_cache:
            apex_indices = self._apex_detector.find_top_k_apex(
                signal=magnitudes, k=self.K_APEX,
            )
            phases = self._apex_detector.find_phase(
                signal=magnitudes,
                apex_indices=apex_indices,
                cutoff_ratio=0.30,
            )
            self._apex_cache[npy_path] = (apex_indices, phases)

        apex_indices, phases = self._apex_cache[npy_path]

        # Feature extraction
        feat = self._extractor.extract(roi_frames, apex_indices, phases)

        # Augmentation only during training
        if self.mode == "train":
            feat = self._extractor.augment(feat)

        return (
            torch.from_numpy(feat),
            torch.tensor(label, dtype=torch.long),
            subject,
        )

    # ── public utilities ──────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Switch between train/eval mode without rebuilding the dataset.

        Args:
            mode: ``"train"`` or ``"eval"``.

        Raises:
            ValueError: If *mode* is not recognised.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"'mode' must be one of {_VALID_MODES}, got '{mode}'"
            )
        self.mode = mode

    def clear_cache(self) -> None:
        """Clear the apex cache.  Call at the start of each new fold."""
        self._apex_cache.clear()

    def get_labels(self) -> np.ndarray:
        """Return all labels as an int array of shape ``(N,)``.

        Useful for ``WeightedRandomSampler`` and distribution analysis.
        """
        return np.array([
            self._parse_label(lv)
            for lv in self.annotations["anxiety_level"]
        ])

    def get_subject_ids(self) -> list[str]:
        """Return all subject IDs.

        Used as the ``groups`` parameter for ``GroupKFold``.
        """
        return self.annotations["subject_id"].tolist()

    def get_all_features(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract all feature vectors without augmentation.

        Useful for PCA analysis and fitting a :class:`Normalizer`
        before training.

        Returns:
            A 3-tuple ``(X, y, subjects)``:

            - ``X``        — ``np.ndarray`` of shape ``(N, 78)``
            - ``y``        — ``np.ndarray`` of shape ``(N,)`` int
            - ``subjects`` — ``list[str]`` of length N
        """
        prev_mode = self.mode
        self.mode = "eval"
        X, y, subs = [], [], []
        for i in range(len(self)):
            feat, lbl, sub = self[i]
            X.append(feat.numpy())
            y.append(lbl.item())
            subs.append(sub)
        self.mode = prev_mode
        return np.array(X, dtype=np.float32), np.array(y, dtype=int), subs

    # ── validation schemes ────────────────────────────────────────────

    @staticmethod
    def build_group_kfold(
        annotations: pd.DataFrame,
        n_splits: int = 5,
    ) -> list[dict]:
        """GroupKFold: all clips from one subject stay in the same fold.

        Prevents subject leakage.  The test set contains approximately
        ``N / k`` subjects per fold.

        Suitable for:
            - Proof of concept that the model can learn expression
              patterns.
            - Stable metrics thanks to a larger test set compared
              to LOPO.

        Args:
            annotations: :attr:`MicroExpressionDataset.annotations`.
            n_splits: Number of folds.

        Returns:
            A list of dicts, each containing keys ``fold``,
            ``train_idx``, ``test_idx``, ``train_subjects``,
            ``test_subjects``, ``train_label_dist``, and
            ``test_label_dist``.

        Example::

            folds = MicroExpressionDataset.build_group_kfold(
                ds.annotations, n_splits=5,
            )
            for fold_info in folds:
                train_sub = Subset(ds, fold_info["train_idx"])
                test_sub  = Subset(ds, fold_info["test_idx"])
        """
        subjects = annotations["subject_id"].tolist()
        labels = annotations["anxiety_level"].apply(
            lambda x: 1 if str(x).strip().lower() == "high" else 0,
        ).tolist()

        gkf = GroupKFold(n_splits=n_splits)
        folds: list[dict] = []

        for fold_idx, (tr, te) in enumerate(
            gkf.split(X=range(len(annotations)), y=labels, groups=subjects),
        ):
            tr_subjs = {annotations.iloc[i]["subject_id"] for i in tr}
            te_subjs = {annotations.iloc[i]["subject_id"] for i in te}
            tr_dist = np.bincount([labels[i] for i in tr], minlength=2)
            te_dist = np.bincount([labels[i] for i in te], minlength=2)

            folds.append({
                "fold": fold_idx,
                "train_idx": tr.tolist(),
                "test_idx": te.tolist(),
                "train_subjects": tr_subjs,
                "test_subjects": te_subjs,
                "train_label_dist": tr_dist,
                "test_label_dist": te_dist,
            })

        n_unique = len(set(subjects))
        logger.info(
            "GroupKFold-%d: %d folds, ~%d subjects per test fold",
            n_splits, len(folds), n_unique // n_splits,
        )
        for f in folds:
            logger.info(
                "  Fold %02d | train=%d dist=%s | test=%d dist=%s",
                f["fold"],
                len(f["train_idx"]), f["train_label_dist"],
                len(f["test_idx"]), f["test_label_dist"],
            )
        return folds

    @staticmethod
    def build_lopo_folds(
        annotations: pd.DataFrame,
        seed: int = 42,
    ) -> list[dict]:
        """LOPO (Leave-One-Pair-Out) validation.

        Each fold removes 1 HIGH subject + 1 LOW subject as the test
        set, ensuring both classes are always present so that UAR/F1
        are meaningful.

        Suitable for:
            - Generalisation claims to unseen subjects.
            - Publication-standard micro-expression recognition
              evaluation.

        Args:
            annotations: :attr:`MicroExpressionDataset.annotations`.
            seed: Random seed for shuffling subjects.

        Returns:
            A list of dicts with keys ``fold``, ``train_idx``,
            ``test_idx``, ``train_subjects``, ``test_subjects``,
            ``train_label_dist``, and ``test_label_dist``.
        """
        subject_df = (
            annotations[["subject_id", "anxiety_level"]]
            .drop_duplicates("subject_id")
            .copy()
        )
        subject_df["anxiety_level"] = (
            subject_df["anxiety_level"].astype(str).str.strip().str.lower()
        )

        high_subjs = subject_df[
            subject_df["anxiety_level"] == "high"
        ]["subject_id"].tolist()
        low_subjs = subject_df[
            subject_df["anxiety_level"] == "low"
        ]["subject_id"].tolist()

        rng = np.random.default_rng(seed)
        rng.shuffle(high_subjs)
        rng.shuffle(low_subjs)

        n_folds = min(len(high_subjs), len(low_subjs))
        all_subjs = set(subject_df["subject_id"])
        label_map = dict(zip(
            subject_df["subject_id"],
            subject_df["anxiety_level"].apply(
                lambda x: 1 if x == "high" else 0,
            ),
        ))

        folds: list[dict] = []
        for i in range(n_folds):
            te_subjs = {high_subjs[i], low_subjs[i]}
            tr_subjs = all_subjs - te_subjs

            te_idx = annotations.index[
                annotations["subject_id"].isin(te_subjs)
            ].tolist()
            tr_idx = annotations.index[
                annotations["subject_id"].isin(tr_subjs)
            ].tolist()

            te_labels = [label_map[annotations.loc[j, "subject_id"]] for j in te_idx]
            tr_labels = [label_map[annotations.loc[j, "subject_id"]] for j in tr_idx]

            folds.append({
                "fold": i,
                "train_idx": tr_idx,
                "test_idx": te_idx,
                "train_subjects": tr_subjs,
                "test_subjects": te_subjs,
                "train_label_dist": np.bincount(tr_labels, minlength=2),
                "test_label_dist": np.bincount(te_labels, minlength=2),
            })

        logger.info(
            "LOPO: %d folds (%d HIGH + %d LOW subjects). "
            "Test per fold: 1 HIGH + 1 LOW (both classes always present)",
            n_folds, len(high_subjs), len(low_subjs),
        )
        return folds

    @staticmethod
    def build_combined_folds(
        annotations: pd.DataFrame,
        n_splits: int = 5,
        seed: int = 42,
    ) -> dict[str, list[dict]]:
        """Combined validation: GroupKFold + LOPO.

        Produces both fold sets at once for direct comparison.
        GroupKFold measures expression-pattern discrimination;
        LOPO measures generalisation to unseen subjects.

        Args:
            annotations: :attr:`MicroExpressionDataset.annotations`.
            n_splits: Number of GroupKFold folds.
            seed: Random seed for LOPO subject shuffling.

        Returns:
            A dict with keys ``"group_kfold"`` and ``"lopo"``, each
            mapping to a list of fold dicts.
        """
        logger.info("=" * 55)
        logger.info("Combined Validation: GroupKFold + LOPO")
        logger.info("=" * 55)

        gkf_folds = MicroExpressionDataset.build_group_kfold(
            annotations, n_splits=n_splits,
        )
        lopo_folds = MicroExpressionDataset.build_lopo_folds(
            annotations, seed=seed,
        )

        return {
            "group_kfold": gkf_folds,
            "lopo": lopo_folds,
        }

    # ── private ───────────────────────────────────────────────────────

    @staticmethod
    def _scan_missing(
        annotations: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Partition annotations into valid and skipped rows.

        Scans every row once so that :meth:`__getitem__` never needs
        to check file existence.

        Args:
            annotations: Raw annotation DataFrame.

        Returns:
            A 2-tuple ``(valid_df, skipped_df)`` where *valid_df*
            contains rows whose ``.npy`` file exists and *skipped_df*
            contains rows that were dropped.
        """
        missing_mask: list[str] = []

        for _, row in annotations.iterrows():
            path = str(row.get("npy_path", ""))
            if not path or not os.path.isabs(path):
                missing_mask.append("path_not_absolute")
            elif not os.path.exists(path):
                missing_mask.append("file_not_found")
            else:
                missing_mask.append("")

        annotations = annotations.copy()
        annotations["_skip_reason"] = missing_mask

        skipped_df = annotations[annotations["_skip_reason"] != ""].copy()
        valid_df = annotations[annotations["_skip_reason"] == ""].drop(
            columns=["_skip_reason"],
        ).reset_index(drop=True)

        n_total = len(annotations)
        n_skipped = len(skipped_df)
        n_valid = len(valid_df)

        if n_skipped > 0:
            logger.warning(
                ".npy file scan: %d/%d valid, %d skipped",
                n_valid, n_total, n_skipped,
            )
            for _, row in skipped_df.iterrows():
                logger.warning(
                    "  SKIP [%s] subject=%s clip=%s stage=%s -> %s",
                    row["_skip_reason"],
                    row.get("subject_id", "?"),
                    row.get("clip", "?"),
                    row.get("stage", "?"),
                    row.get("npy_path", "(no path)"),
                )
        else:
            logger.info("All %d .npy files found.", n_total)

        return valid_df, skipped_df.drop(
            columns=["_skip_reason"], errors="ignore",
        ).reset_index(drop=True)

    @staticmethod
    def _parse_label(anxiety_level: str) -> int:
        """Convert an anxiety-level string to a numeric label.

        Args:
            anxiety_level: Value from the ``anxiety_level`` CSV column.

        Returns:
            ``1`` for ``"high"``, ``0`` for ``"low"``.

        Raises:
            ValueError: If the value is not recognised.
        """
        level = str(anxiety_level).strip().lower()
        if level == "high":
            return 1
        if level == "low":
            return 0
        raise ValueError(
            f"Unrecognised anxiety_level '{anxiety_level}'. "
            f"Expected 'high' or 'low'."
        )
