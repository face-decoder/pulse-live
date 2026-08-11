#!/usr/bin/env python3
"""Inference script for video and optical flow following notebook 0401 schema.

Notebook Reference: combinations-notebooks/0401-onset-apex-behavior-cnn-spasial-temporal.ipynb

Pipeline Schema:
1. Model: SpatioTemporalCNN(in_channels=10, num_classes=2)
   - Operates directly on 5-ROI optical flow tensors (N_roi=5 x C=2 channels = 10 in_channels).
2. Spotting: ApexWindowDetector(percentile=95, prominence=0.5, max_window=512)
3. Window Slicing: Slices 'onset' and 'apex' frames (PHASES = ["onset", "apex"])
4. Transforms: PadAndMask(max_len=512), AugmentFlow(training=False)
5. Test-Time Augmentation (TTA): N_TTA = 8 forward passes with random scale jitter (0.93 - 1.07) and noise (std 0.02)
6. Decision Threshold: Calibrated best_threshold loaded from checkpoint (default ~0.5)

Usage Examples:
--------------
1. Inference on a single video file:
   python inference_0401.py --video path/to/sample.mp4

2. Inference on a pre-computed optical flow NPZ/NPY file:
   python inference_0401.py --npz path/to/flow.npz

3. Batch inference on a directory of video files:
   python inference_0401.py --video-dir path/to/videos_dir/ --output-json results.json

4. Replicate test set evaluation from Notebook 0401 Cell 9:
   python inference_0401.py --annotation /home/inadio/datasets/dataset_test/clips-annotations-v10.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project Imports
from src.apex.modules.apex_phase_spotter_roi import ApexPhaseSpotterROI
from src.dataset.modules.augment_flow import AugmentFlow
from src.dataset.modules.compose import Compose
from src.dataset.modules.flow_roi_dataset import FlowROIDataset
from src.dataset.modules.temporal_transforms import PadAndMask
from src.dataset.modules.window_selector import ApexWindowDetector, WindowSelector
from src.models.modules.spatio_temporal.spatio_temporal_cnn import SpatioTemporalCNN
from src.video.modules import Video

# ── Notebook 0401 Schema Constants ───────────────────────────────────────────
MAX_SEQ_LEN = 512
DETECTOR_PERCENTILE = 95
DETECTOR_PROMINENCE = 0.5
PHASES = ["onset", "apex"]
TARGET_NAMES = ["Anxiety Rendah", "Anxiety Tinggi"]
LABEL_MAP = {"anxiety_rendah": 0, "anxiety_tinggi": 1}
REVERSE_LABEL_MAP = {0: "anxiety_rendah", 1: "anxiety_tinggi"}
N_TTA = 8
USE_TTA_INFERENCE = True
BATCH_SIZE = 8
DEFAULT_CHECKPOINT_DIR = (
    PROJECT_ROOT / "checkpoints_0401-onset-apex-behavior-cnn-spasial-temporal"
)
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_CHECKPOINT_DIR / "best_model.pt"
    if (DEFAULT_CHECKPOINT_DIR / "best_model.pt").exists()
    else PROJECT_ROOT / "combinations-notebooks" / "checkpoints_0401-onset-apex-behavior-cnn-spasial-temporal" / "best_model.pt"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inference_0401")


# ── TTA Helper (Exact implementation from Notebook 0401 Cell 0) ───────────────
def tta_predict_positive_proba(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_mask: Optional[torch.Tensor] = None,
    n_tta: int = N_TTA,
) -> torch.Tensor:
    """Predict positive class probability using Test-Time Augmentation (TTA).

    Applies random scaling (0.93 - 1.07) and Gaussian noise (std 0.02) across n_tta passes.
    """
    model.eval()
    probs = 0.0
    with torch.no_grad():
        for _ in range(n_tta):
            scale = torch.empty(batch_x.size(0), device=batch_x.device).uniform_(
                0.93, 1.07
            )
            scale = scale.view(-1, *([1] * (batch_x.ndim - 1)))
            x_aug = batch_x * scale
            x_aug = x_aug + torch.randn_like(x_aug) * 0.02
            if batch_mask is not None:
                logits = model(x_aug, mask=batch_mask)
            else:
                logits = model(x_aug)
            probs = probs + torch.softmax(logits, dim=1)[:, 1]
    return probs / float(n_tta)


def evaluate_model_tta(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    threshold: float = 0.5,
    use_tta: bool = True,
    n_tta: int = N_TTA,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader matching notebook evaluate_model function."""
    model.eval()
    total_loss, total_n = 0.0, 0
    y_true, y_prob = [], []
    with torch.inference_mode():
        for batch_x, batch_y, batch_mask in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            logits = model(batch_x, mask=batch_mask)
            total_loss += criterion(logits, batch_y).item() * batch_x.size(0)
            total_n += batch_y.size(0)
            if use_tta:
                pos_prob = tta_predict_positive_proba(
                    model, batch_x, batch_mask, n_tta=n_tta
                )
            else:
                pos_prob = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(batch_y.cpu().numpy().tolist())
            y_prob.extend(pos_prob.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    loss = total_loss / max(total_n, 1)
    acc = float((y_pred == y_true).mean()) if total_n > 0 else 0.0
    return loss, acc, y_true, y_pred, y_prob


# ── Raw Video Flow Extractor ──────────────────────────────────────────────────
def extract_flow_from_video(
    video_path: Union[str, Path],
    tile_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract 5-ROI optical flow array from a raw video file.

    Returns:
        np.ndarray of shape (T, 5, 2, tile_h, tile_w)
    """
    video_path = str(video_path)
    logger.info("Extracting optical flow from video: %s", video_path)
    start_t = time.time()

    spotter = ApexPhaseSpotterROI(tile_size=tile_size)
    video = Video(video_path=video_path)
    video.map(spotter.__process_frame__)

    # Collect per-ROI flow arrays across frames
    # spotter.horizontal_magnitudes and spotter.vertical_magnitudes contain list of dx, dy per ROI
    roi_flows_list = []
    for roi_name, _ in spotter.roi_defs:
        dx_list = spotter.horizontal_magnitudes[roi_name]
        dy_list = spotter.vertical_magnitudes[roi_name]
        if not dx_list or not dy_list:
            raise RuntimeError(
                f"Failed to extract optical flow for ROI '{roi_name}' in video: {video_path}"
            )
        # Stack temporal frames for this ROI -> shape (T, 2, H, W)
        dx_arr = np.stack(dx_list, axis=0)  # (T, H, W)
        dy_arr = np.stack(dy_list, axis=0)  # (T, H, W)
        roi_flow = np.stack([dx_arr, dy_arr], axis=1)  # (T, 2, H, W)
        roi_flows_list.append(roi_flow)

    # Stack all 5 ROIs -> shape (T, 5, 2, H, W)
    flow = np.stack(roi_flows_list, axis=1).astype(np.float32)
    elapsed = time.time() - start_t
    logger.info(
        "Extracted flow shape: %s (T=%d frames, elapsed=%.2fs)",
        flow.shape,
        flow.shape[0],
        elapsed,
    )
    return flow


def load_flow_from_file(file_path: Union[str, Path]) -> np.ndarray:
    """Load optical flow array from .npz or .npy file."""
    file_path = Path(file_path)
    data = np.load(file_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data as npz:
            if "flow" not in npz:
                raise KeyError(f"'flow' key not found in {file_path}")
            flow = npz["flow"].astype(np.float32)
    elif isinstance(data, np.ndarray) and data.dtype == object:
        obj = data.item()
        flow = obj["flow"].astype(np.float32)
    elif isinstance(data, np.ndarray):
        flow = data.astype(np.float32)
    else:
        raise ValueError(f"Unrecognized flow format in file: {file_path}")

    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError(f"Expected flow shape (T, N_roi=5, 2, H, W), got {flow.shape}")
    return flow


# ── Inferencer Class for Notebook 0401 Schema ────────────────────────────────
class SpatioTemporal0401Inferencer:
    """Inference engine following notebook 0401 (onset+apex behavior CNN spatial-temporal) schema."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        threshold: Optional[float] = None,
        n_tta: int = N_TTA,
    ) -> None:
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        )
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_tta = int(n_tta)

        # Initialize SpatioTemporalCNN model (in_channels=10, num_classes=2)
        self.model = SpatioTemporalCNN(in_channels=10, num_classes=2).to(self.device)
        self.best_threshold = 0.5

        self._load_checkpoint(threshold)

        # Detector matching notebook 0401
        self.detector = ApexWindowDetector(
            percentile=DETECTOR_PERCENTILE,
            prominence=DETECTOR_PROMINENCE,
            max_window=MAX_SEQ_LEN,
        )

    def _load_checkpoint(self, custom_threshold: Optional[float] = None) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}. "
                f"Please verify path or train the model using notebook 0401."
            )

        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in ck:
            self.model.load_state_dict(ck["model_state_dict"])
        else:
            self.model.load_state_dict(ck)

        self.model.eval()

        if custom_threshold is not None:
            self.best_threshold = float(custom_threshold)
        elif "best_threshold" in ck:
            self.best_threshold = float(ck["best_threshold"])

        logger.info(
            "Model ready on %s | best_threshold=%.3f | n_tta=%d",
            self.device,
            self.best_threshold,
            self.n_tta,
        )

    def predict_flow(self, flow: np.ndarray) -> Dict[str, Any]:
        """Run 0401 inference pipeline on a pre-loaded optical flow tensor.

        Args:
            flow: np.ndarray of shape (T, N_roi=5, 2, H, W)

        Returns:
            Dict containing label, target_name, prob_high, prob_low, confidence,
            best_threshold, n_windows, latencies, etc.
        """
        t_start = time.time()

        # 1. Spot apex micro-expression windows
        t_spot_start = time.time()
        windows, meta = self.detector.detect_windows(flow, phase_mode="full")
        spotting_latency_ms = (time.time() - t_spot_start) * 1000.0
        n_windows = len(windows)

        warning: Optional[str] = None
        if not meta.get("valid", False) or n_windows == 0:
            warning = "No micro-expression window detected; using whole clip fallback."
            T = flow.shape[0]
            apex = T // 2
            windows = [(0, apex, T)]

        # 2. Slice onset & apex frames according to PHASES = ["onset", "apex"]
        slices = []
        for left, apex, right in windows:
            parts = []
            if "onset" in PHASES and apex > left:
                parts.append(flow[left:apex])
            if "apex" in PHASES:
                parts.append(flow[apex : apex + 1])
            if parts:
                slices.append(np.concatenate(parts, axis=0))

        if not slices:
            slices = [flow[:1]]

        merged = np.concatenate(slices, axis=0)[:MAX_SEQ_LEN]
        T_clipped, N_roi, C, H, W = merged.shape

        # 3. Reshape flow (T, 5, 2, H, W) -> tensor shape (1, 10, T, H, W)
        x_tensor = (
            torch.from_numpy(merged.astype(np.float32))
            .permute(1, 2, 0, 3, 4)  # (N_roi, C, T, H, W)
            .reshape(N_roi * C, T_clipped, H, W)  # (10, T, H, W)
            .unsqueeze(0)  # (1, 10, T, H, W)
            .to(self.device)
        )

        # 4. Model TTA forward pass
        t_infer_start = time.time()
        prob_high = float(
            tta_predict_positive_proba(
                self.model, x_tensor, batch_mask=None, n_tta=self.n_tta
            ).item()
        )
        inference_latency_ms = (time.time() - t_infer_start) * 1000.0
        prob_low = 1.0 - prob_high

        # 5. Threshold decision
        label_idx = int(prob_high >= self.best_threshold)
        predicted_label = REVERSE_LABEL_MAP[label_idx]
        target_name = TARGET_NAMES[label_idx]
        confidence = prob_high if label_idx == 1 else prob_low
        total_latency_ms = (time.time() - t_start) * 1000.0

        return {
            "label_idx": label_idx,
            "predicted_label": predicted_label,
            "target_name": target_name,
            "prob_high": prob_high,
            "prob_low": prob_low,
            "confidence": confidence,
            "threshold": self.best_threshold,
            "n_windows": n_windows,
            "clipped_frames": T_clipped,
            "warning": warning,
            "spotting_latency_ms": spotting_latency_ms,
            "inference_latency_ms": inference_latency_ms,
            "total_latency_ms": total_latency_ms,
        }

    def predict_video(
        self, video_path: Union[str, Path], tile_size: Tuple[int, int] = (64, 64)
    ) -> Dict[str, Any]:
        """Run end-to-end inference on a raw video file."""
        flow = extract_flow_from_video(video_path, tile_size=tile_size)
        res = self.predict_flow(flow)
        res["video_path"] = str(video_path)
        return res

    def predict_npz(self, npz_path: Union[str, Path]) -> Dict[str, Any]:
        """Run inference on a pre-computed optical flow .npz or .npy file."""
        flow = load_flow_from_file(npz_path)
        res = self.predict_flow(flow)
        res["npz_path"] = str(npz_path)
        return res

    def run_test_inference(
        self,
        annotation_path: Union[str, Path],
        output_tag: str = "test_results",
        threshold: Optional[float] = None,
        title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Replicate run_test_inference function from Cell 9 of Notebook 0401."""
        from sklearn.metrics import (
            balanced_accuracy_score,
            classification_report,
            f1_score,
            recall_score,
        )

        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            logger.warning("[Skip] Annotation file not found: %s", annotation_path)
            return None

        df_test = pd.read_csv(annotation_path)
        if "npy_path" not in df_test.columns and "cache_path" in df_test.columns:
            df_test["npy_path"] = df_test["cache_path"]
        if "is_valid" in df_test.columns:
            df_test = df_test[df_test["is_valid"]].copy()

        df_test = df_test[df_test["label"].isin(LABEL_MAP)].copy()
        df_test["label_idx"] = df_test["label"].map(LABEL_MAP)
        logger.info(
            "[Test] %d clips, %d subjects",
            len(df_test),
            df_test["subject_id"].nunique(),
        )

        eval_transform = Compose(
            [
                WindowSelector(phase_includes=PHASES),
                PadAndMask(max_len=MAX_SEQ_LEN),
                AugmentFlow(training=False),
            ]
        )

        def _collate_fn(batch):
            xs = torch.stack([item.x for item in batch])
            ys = torch.stack([item.y for item in batch])
            masks = (
                torch.stack([item.mask for item in batch])
                if batch[0].mask is not None
                else torch.zeros(len(batch), xs.shape[-1], dtype=torch.bool)
            )
            return xs, ys, masks

        ds = FlowROIDataset(
            metadata_df=df_test,
            detector=self.detector,
            phase_mode="full",
            transform=eval_transform,
        )
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_fn,
        )

        effective_thr = (
            float(threshold) if threshold is not None else self.best_threshold
        )
        logger.info("[Checkpoint] Using threshold=%.3f", effective_thr)

        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, y_true_t, y_pred_t, y_prob_t = evaluate_model_tta(
            self.model,
            loader,
            criterion,
            threshold=effective_thr,
            use_tta=True,
            n_tta=self.n_tta,
            device=self.device,
        )

        f1_macro = float(f1_score(y_true_t, y_pred_t, average="macro", zero_division=0))
        f1_w = float(f1_score(y_true_t, y_pred_t, average="weighted", zero_division=0))
        bacc = float(balanced_accuracy_score(y_true_t, y_pred_t))
        rec_rd = float(recall_score(y_true_t, y_pred_t, pos_label=0, zero_division=0))
        rec_tg = float(recall_score(y_true_t, y_pred_t, pos_label=1, zero_division=0))

        display_name = title or annotation_path.name
        print("\n" + "=" * 60)
        print(f"[Test Inference — {display_name}]")
        print("=" * 60)
        print(f"  Test Loss        : {test_loss:.4f}")
        print(f"  Test Acc         : {test_acc:.4f}")
        print(f"  Macro F1         : {f1_macro:.4f}")
        print(f"  Weighted F1      : {f1_w:.4f}")
        print(f"  Balanced Acc     : {bacc:.4f}")
        print(f"  Recall rendah    : {rec_rd:.4f}")
        print(f"  Recall tinggi    : {rec_tg:.4f}")
        print()
        print(
            classification_report(
                y_true_t, y_pred_t, target_names=TARGET_NAMES, zero_division=0
            )
        )

        metrics = {
            "annotation_test": str(annotation_path),
            "n_clips": len(df_test),
            "n_subjects": int(df_test["subject_id"].nunique()),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_w,
            "test_bacc": bacc,
            "test_recall_rendah": rec_rd,
            "test_recall_tinggi": rec_tg,
            "best_threshold": effective_thr,
        }

        output_dir = self.checkpoint_path.parent
        metrics_path = output_dir / f"{output_tag}.json"
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2, default=float)
        logger.info("[Saved] %s", metrics_path)

        rows = []
        for yt, yp, ypr in zip(y_true_t, y_pred_t, y_prob_t):
            rows.append(
                {
                    "true_label": int(yt),
                    "true_class": TARGET_NAMES[int(yt)],
                    "predicted_label": TARGET_NAMES[int(yp)],
                    "prob_tinggi": float(ypr),
                    "prob_rendah": float(1 - ypr),
                }
            )
        df_res = pd.DataFrame(rows)
        csv_path = output_dir / f"{output_tag}.csv"
        df_res.to_csv(csv_path, index=False)
        logger.info("[Saved] %s (%d samples)", csv_path, len(df_res))

        return metrics


# ── CLI Entry Point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference using SpatioTemporalCNN (Notebook 0401 Schema)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to input raw video file (.mp4, .avi, .mov)",
    )
    input_group.add_argument(
        "--npz",
        "--npy",
        type=str,
        dest="npz",
        help="Path to pre-computed optical flow file (.npz or .npy)",
    )
    input_group.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing video files for batch processing",
    )
    input_group.add_argument(
        "--annotation",
        type=str,
        help="Path to CSV annotation file for dataset test evaluation (matches Notebook 0401 Cell 9)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to model checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (defaults to best_threshold from checkpoint)",
    )
    parser.add_argument(
        "--n-tta",
        type=int,
        default=N_TTA,
        help="Number of TTA passes during inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run model on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save output prediction JSON",
    )

    args = parser.parse_args()

    inferencer = SpatioTemporal0401Inferencer(
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        n_tta=args.n_tta,
    )

    # 1. CSV Annotation test evaluation mode
    if args.annotation:
        inferencer.run_test_inference(
            annotation_path=args.annotation,
            output_tag="test_results_0401",
            threshold=args.threshold,
        )
        return

    # 2. Single video file
    if args.video:
        res = inferencer.predict_video(args.video)
        results = [res]

    # 3. Pre-computed optical flow file
    elif args.npz:
        res = inferencer.predict_npz(args.npz)
        results = [res]

    # 4. Directory of videos
    elif args.video_dir:
        vdir = Path(args.video_dir)
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_files = [
            p for p in sorted(vdir.rglob("*")) if p.suffix.lower() in video_extensions
        ]
        logger.info("Found %d videos in %s", len(video_files), vdir)
        results = []
        for vp in video_files:
            try:
                res = inferencer.predict_video(vp)
                results.append(res)
            except Exception as exc:
                logger.error("Failed to process video %s: %s", vp, exc)

    # Output printing & saving
    print("\n" + "=" * 60)
    print(" INFERENCE RESULTS (Notebook 0401 Schema — SpatioTemporalCNN)")
    print("=" * 60)
    for r in results:
        target_file = r.get("video_path") or r.get("npz_path") or "Sample"
        print(f"\nFile               : {target_file}")
        print(f"Predicted Class    : {r['target_name']} ({r['predicted_label']})")
        print(f"Prob (High)        : {r['prob_high']:.4f}")
        print(f"Prob (Low)         : {r['prob_low']:.4f}")
        print(f"Confidence         : {r['confidence']:.4f}")
        print(f"Threshold          : {r['threshold']:.3f}")
        print(f"Windows Detected   : {r['n_windows']}")
        print(f"Clipped Frames     : {r['clipped_frames']}")
        print(f"Inference Latency  : {r['inference_latency_ms']:.2f} ms")
        if r.get("warning"):
            print(f"Warning            : {r['warning']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(
                results if len(results) > 1 else results[0],
                fp,
                indent=2,
                default=float,
            )
        logger.info("[Saved] Output JSON written to: %s", out_path)


if __name__ == "__main__":
    main()
