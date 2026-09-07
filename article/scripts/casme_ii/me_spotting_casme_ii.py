import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from torch import nn, optim

from article.src import FrameSequence, MELabel, MESpotting, RealTimeProfiler
from src.dataset.modules.behavioral_features import BehavioralFeatures
from src.models.modules.spatio_temporal.spatio_temporal_cnn import SpatioTemporalCNN

# 1. Load CASME II annotations & initialize utilities
CASME2_DIR = (
    "/home/inadio/datasets/secondaries/casme-ii"
    if os.path.exists("/home/inadio/datasets/secondaries/casme-ii")
    else "/home/inadio/datasets/secondaries/casme-2"
)
ANNOTATIONS_PATH = Path(CASME2_DIR) / "annotations.xlsx"
CACHE_DIR = Path(CASME2_DIR) / "cache"

df = pd.read_excel(ANNOTATIONS_PATH)
df = df[(df["OffsetFrame"] - df["OnsetFrame"] + 1) <= 100].copy()
print(f"Loaded and filtered annotations to {len(df)} micro-expression entries.")


FPS = 200
MAX_SEQUENCE_LENGTH = 100
CNN_MAX_LEN = 64
BATCH_SIZE = 8
CNN_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def section(title):
    print(f"\n## {title}")


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def show_metrics(title, values):
    section(title)
    print(pd.DataFrame([values]).to_string(index=False, float_format="%.4f"))


def compute_iou(interval_a, interval_b):
    # Matches Fang et al. 2023 (RMES) / MESpotting.compute_iou convention:
    # interval measure is (end - start), not inclusive frame count.
    s_on, s_off = interval_a
    g_on, g_off = interval_b
    intersection = max(0, min(s_off, g_off) - max(s_on, g_on))
    union = (s_off - s_on) + (g_off - g_on) - intersection
    return float(intersection / union) if union > 0 else 0.0


def load_clipped(npz_path, feat_interval):
    seq = FrameSequence.from_npz(npz_path, fps=FPS)
    if seq is None or len(seq) == 0:
        return None
    return seq.clip(*feat_interval)


spotter = MESpotting(
    cutoff_ratio=0.20, fps=FPS, smooth_window_ms=300, thresh_window_ms=900
)
extractor = BehavioralFeatures()
profiler = RealTimeProfiler()

sequences: list[FrameSequence] = []
clipped_sequences: list[FrameSequence] = []
all_features = []
labels = []
groups = []
spotted_intervals = []
gt_intervals = []

# 2. Extract sequences, spot boundaries & slice micro-expressions
for _, row in df.iterrows():
    subject = row["Subject"]
    subject_str = f"sub{int(subject):02d}" if pd.notnull(subject) else "sub01"
    filename = str(row["Filename"])
    raw_emotion = row["Estimated Emotion"]

    label = MELabel.map(raw_emotion, target="2-class")
    if label is None:
        continue

    npz_path = os.path.join(CACHE_DIR, f"{subject_str}_{filename}.npz")
    seq = FrameSequence.from_npz(
        npz_path, subject=subject_str, clip_name=filename, label=label, fps=FPS
    )
    if seq is None or len(seq) == 0:
        continue

    gt_interval = (int(row["OnsetFrame"]), int(row["OffsetFrame"]))

    # Best-IoU match against gt (standard detection-eval assignment), not
    # spot()'s arbitrary first-detected-candidate selection.
    candidates = spotter.spot_all(seq.magnitudes)
    match_result = spotter.match(candidates, gt_interval, fallback_mags=seq.magnitudes)
    feat_interval = match_result.get("feat_interval", gt_interval)
    spot_interval = match_result.get("spot_interval", gt_interval)

    clipped_seq = seq.clip(feat_interval[0], feat_interval[1])
    if len(clipped_seq) == 0:
        continue

    sequences.append(seq)
    clipped_sequences.append(clipped_seq)
    spotted_intervals.append(spot_interval)
    gt_intervals.append(gt_interval)
    labels.append(label)
    groups.append(subject_str)

    flow_tensor = torch.from_numpy(clipped_seq.flow)
    feats = extractor._extract(flow_tensor).cpu().numpy()
    all_features.append(feats)

# 3. Spotting Performance Benchmark (Fang et al. 2023 protocol)
spot_metrics = spotter.benchmark(spotted_intervals, gt_intervals, iou_thresh=0.5)
print("\n=== Spotting Performance (CASME II | Fang et al. 2023 Standard) ===")
print(f"Total Samples:       {spot_metrics['samples']}")
print(f"True Positives:      {spot_metrics['tp']} (IoU >= 0.5)")
print(f"Average IoU:         {spot_metrics['mean_iou']:.4f}")
print(f"Spotting Precision:  {spot_metrics['precision']:.4f}")
print(f"Spotting Recall:     {spot_metrics['recall']:.4f}")
print(f"Spotting F1-Score:   {spot_metrics['f1']:.4f}")
print("===================================================================\n")

# 4. Feature Matrix & LOSO Cross-Validation (Baseline SVM)
y, class_names = MELabel.encode(labels)
groups_arr = np.array(groups)
X_static = np.stack(
    [np.concatenate([f.mean(axis=0), f.std(axis=0)]) for f in all_features]
)

logo = LeaveOneGroupOut()
splits = list(logo.split(X_static, y, groups=groups_arr))
print(f"LOSO Cross-Validation | Total Subjects (Splits): {len(splits)}")

svm_preds = np.zeros_like(y)
for train_idx, test_idx in splits:
    X_train, y_train = X_static[train_idx], y[train_idx]
    X_test, y_test = X_static[test_idx], y[test_idx]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(
        kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    svm_preds[test_idx] = clf.predict(X_test_scaled)

show_metrics(
    "CASME II SVM Classification Performance (LOSO)",
    classification_metrics(y, svm_preds),
)

# 5. SpatioTemporalCNN 3D Deep Learning Classification (LOSO)
print(f"\nEvaluating SpatioTemporalCNN 3D Deep Learning model on device: {DEVICE}...")

cnn_preds = np.zeros_like(y)

for fold_idx, (train_idx, test_idx) in enumerate(splits):
    train_seqs = [clipped_sequences[i] for i in train_idx]
    test_seqs = [clipped_sequences[i] for i in test_idx]
    y_train = torch.tensor(y[train_idx], dtype=torch.long, device=DEVICE)

    model = SpatioTemporalCNN(
        in_channels=10, num_classes=len(class_names), dropout_p=0.3
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(CNN_EPOCHS):
        perm = torch.randperm(len(train_seqs))
        for i in range(0, len(train_seqs), BATCH_SIZE):
            indices = perm[i : i + BATCH_SIZE]
            if len(indices) < 2:
                continue
            batch_seqs = [train_seqs[idx] for idx in indices]
            batch_x = FrameSequence.pad_batch(
                batch_seqs, max_len=CNN_MAX_LEN, device=DEVICE
            )
            batch_y = y_train[indices]

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_fold_preds = []
        for i in range(0, len(test_seqs), BATCH_SIZE):
            batch_seqs = test_seqs[i : i + BATCH_SIZE]
            batch_x = FrameSequence.pad_batch(
                batch_seqs, max_len=CNN_MAX_LEN, device=DEVICE
            )
            out_test = model(batch_x)
            test_fold_preds.append(torch.argmax(out_test, dim=1).cpu().numpy())
        if test_fold_preds:
            cnn_preds[test_idx] = np.concatenate(test_fold_preds)

show_metrics(
    "CASME II SpatioTemporalCNN Classification Performance (LOSO)",
    classification_metrics(y, cnn_preds),
)
print("Classification Report:")
print(classification_report(y, cnn_preds, target_names=class_names, zero_division=0))

# 6. Real-Time End-to-End Latency Benchmarking with RealTimeProfiler
print("\n=== Running Real-Time Performance Profiling ===")
scaler_full = RobustScaler().fit(X_static)
svm_full = SVC(
    kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42
).fit(scaler_full.transform(X_static), y)

cnn_full = SpatioTemporalCNN(in_channels=10, num_classes=len(class_names)).to(DEVICE)
cnn_full.eval()

frame_counts = [len(s) for s in sequences]

for seq in sequences:
    with profiler.record("spot"):
        feat_int, _ = spotter.spot(seq.magnitudes, fallback_half_win=49)

    clipped = seq.clip(feat_int[0], feat_int[1])
    if len(clipped) == 0:
        continue

    with profiler.record("cnn_infer"):
        x_cnn = clipped.to_tensor(device=DEVICE)
        with torch.no_grad():
            _ = cnn_full(x_cnn)

section("Real-Time Performance Statistics")
print(profiler.report(frame_counts).to_string(index=False))
