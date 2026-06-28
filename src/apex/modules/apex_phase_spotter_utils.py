from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter

from .apex_phase import ApexPhase
from .apex_smoother import ApexSmoother
from src.dataset.constants.index import ROI_ORDER_DEFAULT
from src.face.modules import FaceRoiPoints


def build_roi_defs() -> list[frozenset]:
    return [
        frozenset(FaceRoiPoints.LEFT_EYE_POINTS),
        frozenset(FaceRoiPoints.RIGHT_EYE_POINTS),
        frozenset(FaceRoiPoints.LIPS_POINTS),
        frozenset(FaceRoiPoints.LEFT_EYEBROW_POINTS),
        frozenset(FaceRoiPoints.RIGHT_EYEBROW_POINTS),
    ]


def detect_all_landmarks(video, landmarker, detect_interval: int) -> list:
    n = len(video)
    interval = max(1, int(detect_interval))

    keyframe_indices = list(range(0, n, interval))
    if keyframe_indices[-1] != n - 1:
        keyframe_indices.append(n - 1)

    keyframe_set = set(keyframe_indices)
    keyframe_landmarks = {}

    cap = cv2.VideoCapture(video.video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in keyframe_set:
            keyframe_landmarks[frame_idx] = landmarker.detect(frame)
        frame_idx += 1
    cap.release()

    all_landmarks = [None] * n
    for idx in keyframe_indices:
        all_landmarks[idx] = keyframe_landmarks[idx]

    for k in range(len(keyframe_indices) - 1):
        start_idx = keyframe_indices[k]
        end_idx = keyframe_indices[k + 1]
        span = max(1, end_idx - start_idx)
        lm_start = keyframe_landmarks[start_idx]
        lm_end = keyframe_landmarks[end_idx]

        for i in range(start_idx + 1, end_idx):
            t = (i - start_idx) / span
            all_landmarks[i] = landmarker.interpolate_landmarks(lm_start, lm_end, t)

    return all_landmarks


def crop_roi_canvas(
    *,
    frame: np.ndarray,
    landmarks,
    aligned_landmarks,
    landmarker,
    aligner,
    roi_defs: Sequence,
    margin: float,
    tile_size: Tuple[int, int],
) -> list[np.ndarray]:
    try:
        aligned = aligner.align(image=frame, landmarks=landmarks)
        aligned_landmarks = landmarker.detect(aligned)
    except Exception:
        aligned = frame
        aligned_landmarks = landmarks if aligned_landmarks is None else aligned_landmarks

    tile_w, tile_h = tile_size
    crops: list[np.ndarray] = []
    for roi_points in roi_defs:
        try:
            roi, _ = landmarker.crop_roi(
                image=aligned,
                landmark_result=aligned_landmarks,
                roi_points=roi_points,
                margin=margin,
                target_size=tile_size,
            )
        except Exception:
            roi = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        crops.append(roi)
    return crops


def crop_fullface(
    *,
    frame: np.ndarray,
    landmarks,
    landmarker,
    margin: float,
    face_size: Tuple[int, int],
) -> np.ndarray:
    try:
        return landmarker.crop(
            image=frame,
            landmarks=landmarks,
            margin=margin,
            output_size=face_size,
        )
    except Exception:
        return np.zeros((face_size[1], face_size[0], 3), dtype=np.uint8)


def flow_to_magnitude_signal(
    flow: np.ndarray,
    selected_rois: Optional[Sequence[str]] = None,
) -> np.ndarray:
    flow = np.asarray(flow)

    if flow.ndim == 5:
        if flow.shape[2] != 2:
            raise ValueError(
                f"ROI flow must have shape (T, N_roi, 2, H, W), got {flow.shape}"
            )
        dx = flow[:, :, 0, :, :].mean(axis=(2, 3))
        dy = flow[:, :, 1, :, :].mean(axis=(2, 3))
        if selected_rois is not None:
            roi_idx = np.array([ROI_ORDER_DEFAULT.index(r) for r in selected_rois])
            dx = dx[:, roi_idx]
            dy = dy[:, roi_idx]
        return np.sqrt(dx**2 + dy**2).mean(axis=1)

    if flow.ndim == 4:
        if flow.shape[-1] == 2:
            dx = flow[..., 0]
            dy = flow[..., 1]
        elif flow.shape[1] == 2:
            dx = flow[:, 0, :, :]
            dy = flow[:, 1, :, :]
        else:
            raise ValueError(
                f"Raw flow must be (T, H, W, 2) or (T, 2, H, W), got {flow.shape}"
            )
        return np.sqrt(dx**2 + dy**2).mean(axis=(1, 2))

    if flow.ndim == 3 and flow.shape[-1] == 2:
        return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean(axis=(1, 2))

    raise ValueError(
        "flow must be raw flow (T, H, W, 2)/(T, 2, H, W) or ROI flow "
        f"(T, N_roi, 2, H, W), got {flow.shape}"
    )


def smooth_signal(signal: Sequence[float]) -> np.ndarray:
    signal_arr = np.asarray(signal, dtype=np.float32)
    if len(signal_arr) < 3:
        return signal_arr
    window_length = ApexSmoother.calculate_window_length(len(signal_arr))
    polyorder = ApexSmoother.calculate_polyorder(window_length)
    return np.asarray(savgol_filter(signal_arr, window_length, polyorder), dtype=np.float32)


def detect_windows_from_signal(
    signal: Sequence[float],
    *,
    percentile: float,
    prominence: float,
    min_distance: int,
    ratio: float,
    min_window: int,
    max_window: int,
    context: int,
    phase_mode: str = "onset_to_apex",
) -> Tuple[List[Tuple[int, int, int]], dict]:
    signal_arr = np.asarray(signal, dtype=np.float32)
    T = int(signal_arr.shape[0])
    if T < 2:
        return [], {"valid": False, "reason": "too_short", "signal_length": T}

    smoothed = smooth_signal(signal_arr)

    apex_phase = ApexPhase(
        distance_threshold=min_distance,
        prominence_threshold=prominence,
        cutoff_ratio=ratio,
    )
    # v6-style height threshold: mean + std (more permissive than percentile)
    height_threshold = float(np.mean(smoothed) + np.std(smoothed))
    # v6-style top-K selection: up to 10 peaks
    peaks = apex_phase.find_top_k_apex(signal=smoothed.tolist(), k=10, height=height_threshold)

    if len(peaks) == 0:
        return [], {"valid": False, "reason": "no_peaks", "signal_length": T}

    phases = apex_phase.find_phase(signal=smoothed.tolist(), apex_indices=peaks, cutoff_ratio=ratio, phase_mode=phase_mode)
    windows: List[Tuple[int, int, int]] = []
    for p in peaks:
        phase = phases.get(p, {})
        left = int(phase.get("start", 0))
        right = int(phase.get("end", 0)) + 1
        if phase_mode == "onset_to_apex":
            right = int(p) + 1

        length = right - left
        if length < min_window:
            continue
        if length > max_window:
            half = max_window // 2
            left = max(0, int(p) - half)
            right = min(T, int(p) + half)

        left = max(0, left - context)
        if phase_mode != "onset_to_apex":
            right = min(T, right + context)

        windows.append((int(left), int(p), int(right)))

    if not windows:
        return [], {
            "valid": False,
            "reason": "no_valid_windows",
            "num_peaks": len(peaks),
            "signal_length": T,
        }

    apex_vals = smoothed[[p for _, p, _ in windows]]
    confidence = float(np.mean(np.abs(apex_vals)) / (np.mean(np.abs(smoothed)) + 1e-6))
    return windows, {
        "valid": True,
        "reason": "ok",
        "num_peaks": len(peaks),
        "num_windows": len(windows),
        "confidence": confidence,
        "signal_length": T,
        "smoothed_signal": smoothed.tolist(),
        "phases": phases,
    }


def stack_roi_flow_canvases(
    flow_items: Iterable[np.ndarray],
    *,
    roi_defs: Sequence,
    tile_size: Tuple[int, int],
) -> np.ndarray:
    frames_5d = []
    n_roi = len(roi_defs)
    tile_w, tile_h = tile_size

    for canvas in flow_items:
        canvas_f = np.asarray(canvas, dtype=np.float32)
        roi_tiles = []
        for j in range(n_roi):
            r, c = divmod(j, 3)
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            tile = canvas_f[y1:y2, x1:x2, :]
            roi_tiles.append(tile.transpose(2, 0, 1))
        frames_5d.append(np.stack(roi_tiles, axis=0))

    return np.stack(frames_5d, axis=0)


def stack_fullface_flows(flow_items: Iterable[np.ndarray]) -> np.ndarray:
    return np.stack(
        [np.asarray(flow, dtype=np.float32).transpose(2, 0, 1) for flow in flow_items],
        axis=0,
    )[:, None, :, :, :]
