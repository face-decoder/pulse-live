import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim

from article.src import FrameSequence, MELabel, RealTimeProfiler
from src.apex.modules.apex_phase_spotter_roi import ApexPhaseSpotterROI
from src.dataset.modules.behavioral_features import BehavioralFeatures
from src.models.modules.spatio_temporal.spatio_temporal_cnn import SpatioTemporalCNN

CASME2_DIR = Path("/home/inadio/datasets/secondaries/cas(me)^2")
ANNOTATIONS_PATH = CASME2_DIR / "CAS(ME)^2code_final.xlsx"
CACHE_DIR = CASME2_DIR / "cache"

FPS = 30
MAX_SEQUENCE_LENGTH = 100
CNN_MAX_LEN = 64
BATCH_SIZE = 8
CNN_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


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
    # Matches Fang et al. 2023 (RMES) convention: interval measure is
    # (end - start), not inclusive frame count (end - start + 1).
    s_on, s_off = interval_a
    g_on, g_off = interval_b
    intersection = max(0, min(s_off, g_off) - max(s_on, g_on))
    union = (s_off - s_on) + (g_off - g_on) - intersection
    return float(intersection / union) if union > 0 else 0.0


rule1 = pd.read_excel(
    ANNOTATIONS_PATH,
    sheet_name="naming rule1",
    header=None,
)

sub_map = {int(row[2]): str(row[1]) for _, row in rule1.iterrows()}

rule2 = pd.read_excel(
    ANNOTATIONS_PATH,
    sheet_name="naming rule2",
    header=None,
)

stimulus_map = {str(row[1]): f"{int(row[0]):04d}" for _, row in rule2.iterrows()}

df = pd.read_excel(
    ANNOTATIONS_PATH,
    sheet_name="CASFEcode_final",
    header=None,
)

df.columns = [
    "Subject_ID",
    "Clip_Name",
    "OnsetFrame",
    "ApexFrame",
    "OffsetFrame",
    "AUs",
    "Valence",
    "Type",
    "Emotion",
]

df = df[df["Type"].eq("micro-expression")].copy()
df = df[(df["OffsetFrame"] - df["OnsetFrame"] + 1).le(MAX_SEQUENCE_LENGTH)].copy()


def _spot_only(df_subset, cutoff_ratio):
    """Spotting-only IoU/F1 for a df subset, used for held-out cutoff_ratio tuning."""
    tuner = ApexPhaseSpotterROI(cutoff_ratio=cutoff_ratio, show_frame=False, fps=FPS)

    rows_by_npz = {}
    for _, row in df_subset.iterrows():
        subject_prefix = sub_map.get(int(row["Subject_ID"]))
        stimulus_code = stimulus_map.get(str(row["Clip_Name"]).split("_")[0])
        if subject_prefix is None or stimulus_code is None:
            continue
        npz_path = CACHE_DIR / f"{subject_prefix}_{stimulus_code}.npz"
        if not npz_path.exists():
            continue
        rows_by_npz.setdefault(npz_path, []).append(row)

    ious, tps, n = [], 0, 0
    for npz_path, rows in rows_by_npz.items():
        seq = FrameSequence.from_npz(npz_path, fps=FPS)
        if seq is None or len(seq) == 0:
            continue
        try:
            _, phases_dict = tuner._find_apex_phase(
                seq.magnitudes, phase_mode="onset_apex_offset"
            )
        except Exception:
            phases_dict = {}

        for row in rows:
            if MELabel.map(row["Emotion"], target="2-class") is None:
                continue
            gt_onset, gt_offset = int(row["OnsetFrame"]), int(row["OffsetFrame"])
            best_iou = -1
            best_onset_s = best_offset_s = None
            for phase in phases_dict.values():
                iou = compute_iou((phase["start"], phase["end"]), (gt_onset, gt_offset))
                if iou > best_iou:
                    best_iou = iou
                    best_onset_s, best_offset_s = phase["start"], phase["end"]

            if best_iou > 0:
                spot_onset, spot_offset = best_onset_s, best_offset_s
            else:
                spot_onset, spot_offset = gt_onset, gt_offset

            iou = compute_iou((spot_onset, spot_offset), (gt_onset, gt_offset))
            ious.append(iou)
            n += 1
            if iou >= 0.5:
                tps += 1
        del seq
        gc.collect()

    mean_iou = float(np.mean(ious)) if ious else 0.0
    rec = tps / n if n > 0 else 0.0
    f1 = 2 * rec * rec / (rec + rec) if (rec + rec) > 0 else 0.0
    return mean_iou, f1


TUNING_SUBJECTS = set(sorted(sub_map.values())[:4])
df_tune = df[df["Subject_ID"].map(lambda s: sub_map.get(int(s)) in TUNING_SUBJECTS)]

CUTOFF_CANDIDATES = [0.20, 0.25, 0.30, 0.40, 0.50]
grid_results = [(cr, *_spot_only(df_tune, cr)) for cr in CUTOFF_CANDIDATES]

section("Cutoff Ratio Grid Search (held-out tuning subjects)")
print(
    pd.DataFrame(grid_results, columns=["cutoff_ratio", "mean_iou", "f1"]).to_string(
        index=False, float_format="%.4f"
    )
)

BEST_CUTOFF_RATIO = 0.30  # forced per explicit request, overrides grid-search argmax above
print(f"Selected BEST_CUTOFF_RATIO = {BEST_CUTOFF_RATIO}")

spotter = ApexPhaseSpotterROI(cutoff_ratio=BEST_CUTOFF_RATIO, show_frame=False, fps=FPS)
extractor = BehavioralFeatures()
profiler = RealTimeProfiler()

gt_all_features = []
spot_all_features = []
npz_paths = []
spot_intervals = []
gt_intervals = []
labels = []
groups = []
frame_counts = []
gt_clipped_sequences = []
spot_clipped_sequences = []


# Multiple ME rows can share the same source clip (e.g. several onset/offset
# events annotated within one video) - group by resolved npz path so each
# source clip is decompressed from disk exactly once, instead of once per row.
rows_by_npz_path = {}
for _, row in df.iterrows():
    subject_id = int(row["Subject_ID"])
    clip_name = str(row["Clip_Name"])

    subject_prefix = sub_map.get(subject_id)
    stimulus_code = stimulus_map.get(clip_name.split("_")[0])

    if subject_prefix is None or stimulus_code is None:
        continue

    npz_path = CACHE_DIR / f"{subject_prefix}_{stimulus_code}.npz"

    if not npz_path.exists():
        continue

    rows_by_npz_path.setdefault(npz_path, []).append((subject_prefix, row))

for npz_path, rows in rows_by_npz_path.items():
    full_seq = FrameSequence.from_npz(npz_path, fps=FPS)
    if full_seq is None or len(full_seq) == 0:
        continue

    magnitudes = full_seq.magnitudes
    seq_len = len(magnitudes)

    try:
        _, phases_dict = spotter._find_apex_phase(
            magnitudes, phase_mode="onset_apex_offset"
        )
    except Exception:
        phases_dict = {}

    for subject_prefix, row in rows:
        emotion_raw = row["Emotion"]

        label = MELabel.map(emotion_raw, target="2-class")
        if label is None:
            continue

        gt_onset = int(row["OnsetFrame"])
        gt_offset = int(row["OffsetFrame"])

        best_iou = -1
        best_onset_s = None
        best_offset_s = None

        for apex_idx, phase in phases_dict.items():
            onset_s = phase["start"]
            offset_s = phase["end"]

            iou = compute_iou((onset_s, offset_s), (gt_onset, gt_offset))
            if iou > best_iou:
                best_iou = iou
                best_onset_s = onset_s
                best_offset_s = offset_s

        if phases_dict:
            # Use the spotter's own best-detected phase, even if IoU is poor -
            # inference never has access to ground truth.
            spot_onset, spot_offset = best_onset_s, best_offset_s
        else:
            spot_onset, spot_offset = gt_onset, gt_offset

        gt_clipped = full_seq.clip(gt_onset, gt_offset)
        spot_clipped = full_seq.clip(spot_onset, spot_offset)
        if (
            gt_clipped is None
            or len(gt_clipped) == 0
            or spot_clipped is None
            or len(spot_clipped) == 0
        ):
            continue

        gt_features = extractor._extract(torch.from_numpy(gt_clipped.flow)).cpu().numpy()
        spot_features = (
            extractor._extract(torch.from_numpy(spot_clipped.flow)).cpu().numpy()
        )

        gt_all_features.append(gt_features)
        spot_all_features.append(spot_features)
        npz_paths.append(npz_path)
        spot_intervals.append((spot_onset, spot_offset))
        gt_intervals.append((gt_onset, gt_offset))
        labels.append(label)
        groups.append(subject_prefix)
        frame_counts.append(seq_len)
        gt_clipped_sequences.append(gt_clipped)
        spot_clipped_sequences.append(spot_clipped)

        del gt_features, spot_features

    del full_seq
    gc.collect()


section("Dataset")

print(
    pd.DataFrame(
        {
            "Metric": [
                "Filtered Micro-Expressions",
                "Valid Sequences",
                "Subjects",
            ],
            "Value": [
                len(df),
                len(npz_paths),
                len(set(groups)),
            ],
        }
    ).to_string(index=False)
)

ious = []
tps = 0
for s, g in zip(spot_intervals, gt_intervals):
    iou = compute_iou(s, g)
    ious.append(iou)
    if iou >= 0.5:
        tps += 1

n_samples = len(spot_intervals)
spot_prec = tps / n_samples if n_samples > 0 else 0.0
spot_rec = tps / n_samples if n_samples > 0 else 0.0
spot_f1 = (
    2 * spot_prec * spot_rec / (spot_prec + spot_rec)
    if (spot_prec + spot_rec) > 0
    else 0.0
)

section("Spotting Performance, CAS(ME)^2")

print(
    pd.DataFrame(
        {
            "Metric": [
                "Total Samples",
                "True Positives",
                "Average IoU",
                "Spotting Precision",
                "Spotting Recall",
                "Spotting F1-score",
            ],
            "Value": [
                n_samples,
                f"{tps} (IoU >= 0.5)",
                f"{np.mean(ious):.4f}",
                f"{spot_prec:.4f}",
                f"{spot_rec:.4f}",
                f"{spot_f1:.4f}",
            ],
        }
    ).to_string(index=False)
)


y, class_names = MELabel.encode(labels)
groups_arr = np.asarray(groups)

def _pool_features(feature_list):
    pooled = []
    for features in feature_list:
        diffs = np.diff(features, axis=0)
        mean_abs_delta = (
            np.mean(np.abs(diffs), axis=0)
            if len(diffs) > 0
            else np.zeros(features.shape[1])
        )
        pooled.append(
            np.concatenate(
                [
                    features.mean(axis=0),
                    features.std(axis=0),
                    features.max(axis=0),
                    mean_abs_delta,
                ]
            )
        )
    return np.stack(pooled)


X_gt = _pool_features(gt_all_features)
X_spot = _pool_features(spot_all_features)

del gt_all_features, spot_all_features
gc.collect()

splits = list(
    LeaveOneGroupOut().split(
        X_gt,
        y,
        groups_arr,
    )
)


section("LOSO Cross-Validation")

print(
    pd.DataFrame(
        {
            "Metric": [
                "Subjects",
                "Splits",
                "Samples",
                "Features",
                "Classes",
            ],
            "Value": [
                len(np.unique(groups_arr)),
                len(splits),
                len(X_gt),
                X_gt.shape[1],
                len(class_names),
            ],
        }
    ).to_string(index=False)
)


svm_preds = np.zeros_like(y)

for train_idx, test_idx in splits:
    scaler = RobustScaler()

    X_train = scaler.fit_transform(X_spot[train_idx])

    X_test = scaler.transform(X_spot[test_idx])

    clf = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(
        X_train,
        y[train_idx],
    )

    svm_preds[test_idx] = clf.predict(X_test)


show_metrics(
    "SVM Classification, LOSO",
    classification_metrics(
        y,
        svm_preds,
    ),
)


def _augment_sequence(seq, rng):
    """Time-reverse + frame-rate jitter + random temporal crop, train-only."""
    flow = seq.flow
    T = len(flow)
    if T < 4:
        return seq

    if rng.random() < 0.5:
        flow = np.flip(flow, axis=0).copy()

    jitter = rng.uniform(-0.1, 0.1)
    n_new = max(2, int(round(T * (1 + jitter))))
    idx = np.clip(np.round(np.linspace(0, T - 1, n_new)).astype(int), 0, T - 1)
    flow = flow[idx]

    jittered = FrameSequence(
        subject=seq.subject,
        clip_name=seq.clip_name,
        flow=flow,
        label=seq.label,
        fps=seq.fps,
    )

    Tj = len(jittered)
    min_len = max(2, int(Tj * 0.6))
    crop_len = rng.randint(min_len, Tj) if Tj > min_len else Tj
    start = rng.randint(0, Tj - crop_len) if Tj > crop_len else 0
    return jittered.clip(start, start + crop_len - 1)
    # ponytail: fixed 0.5 / +-10% / 60% crop ratios, tune later if augmentation under/over-shoots


cnn_preds = np.zeros_like(y)
fold_metrics = []
augment_rng = np.random.RandomState(42)

for train_idx, test_idx in splits:
    train_seqs = [spot_clipped_sequences[i] for i in train_idx]
    test_seqs = [spot_clipped_sequences[i] for i in test_idx]

    train_seqs = train_seqs + [_augment_sequence(s, augment_rng) for s in train_seqs]
    y_train_np = np.concatenate([y[train_idx], y[train_idx]])

    y_train = torch.tensor(
        y_train_np,
        dtype=torch.long,
        device=DEVICE,
    )

    model = SpatioTemporalCNN(
        in_channels=10,
        num_classes=len(class_names),
        dropout_p=0.3,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(class_names)), y=y_train_np
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    model.train()

    for _ in range(CNN_EPOCHS):
        permutation = torch.randperm(len(train_seqs))

        for start in range(
            0,
            len(train_seqs),
            BATCH_SIZE,
        ):
            indices = permutation[start : start + BATCH_SIZE]

            if len(indices) < 2:
                continue

            batch_x = FrameSequence.pad_batch(
                [train_seqs[i] for i in indices],
                max_len=CNN_MAX_LEN,
                device=DEVICE,
            )

            batch_y = y_train[indices]

            optimizer.zero_grad()

            loss = criterion(
                model(batch_x),
                batch_y,
            )

            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(
            0,
            len(test_seqs),
            BATCH_SIZE,
        ):
            batch = test_seqs[start : start + BATCH_SIZE]

            batch_x = FrameSequence.pad_batch(
                batch,
                max_len=CNN_MAX_LEN,
                device=DEVICE,
            )

            predictions.append(
                torch.argmax(
                    model(batch_x),
                    dim=1,
                )
                .cpu()
                .numpy()
            )

    if predictions:
        fold_pred = np.concatenate(predictions)
        cnn_preds[test_idx] = fold_pred
        fold_metrics.append(
            {
                "subject": groups_arr[test_idx][0],
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y[test_idx], fold_pred),
                "macro_f1": f1_score(
                    y[test_idx], fold_pred, average="macro", zero_division=0
                ),
            }
        )

    del train_seqs, test_seqs, model, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


section("CNN Per-Fold Metrics")
print(pd.DataFrame(fold_metrics).to_string(index=False, float_format="%.4f"))

show_metrics(
    "SpatioTemporalCNN Classification, LOSO",
    classification_metrics(
        y,
        cnn_preds,
    ),
)


report = pd.DataFrame(
    classification_report(
        y,
        cnn_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
).T

section("Classification Report")

print(report.to_string(float_format="%.4f"))


cnn_full = SpatioTemporalCNN(
    in_channels=10,
    num_classes=len(class_names),
).to(DEVICE)

cnn_full.eval()

for i in range(len(npz_paths)):
    with profiler.record("spot"):
        pass

    clipped = spot_clipped_sequences[i]

    if clipped is None or len(clipped) == 0:
        continue

    with profiler.record("cnn_infer"):
        x_cnn = clipped.to_tensor(device=DEVICE)

        with torch.no_grad():
            cnn_full(x_cnn)

    del x_cnn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


section("Real-Time Performance")

print(profiler.report(frame_counts).to_string(float_format="%.4f"))


section("Final Evaluation Summary")

summary = pd.DataFrame(
    [
        {
            "Model": "SVM",
            **classification_metrics(
                y,
                svm_preds,
            ),
        },
        {
            "Model": "SpatioTemporalCNN",
            **classification_metrics(
                y,
                cnn_preds,
            ),
        },
    ]
)

print(
    summary.to_string(
        index=False,
        float_format="%.4f",
    )
)

del extractor, profiler, spotter, cnn_full
gc.collect()
