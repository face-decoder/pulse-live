import math
import numpy as np
import cv2
from typing import List, Tuple, Literal, Optional, Union
from scipy.signal import savgol_filter

from src.video.modules import Video, LazyVideo
from src.face.modules import FaceLandmark, FaceRoiPoints
from src.optical_flow.modules import TVL1
from src.apex.modules import ApexPhase, ApexSmoother


ExtractionMode = Literal["roi", "fullface"]


class _LazyFrames:
    """Helper class untuk akses lazy ke frame yang sudah dicrop/preprocess."""
    def __init__(self, lazy_video: LazyVideo, landmarks: List, mode: str,
                 landmarker: FaceLandmark, roi_defs: List, margin: float,
                 tile_size: Tuple[int, int], face_size: Tuple[int, int]):
        self.video = lazy_video
        self.landmarks = landmarks
        self.mode = mode
        self.landmarker = landmarker
        self.roi_defs = roi_defs
        self.margin = margin
        self.tile_size = tile_size
        self.face_size = face_size

        self.cols = 3
        self.rows = math.ceil(len(roi_defs) / self.cols)
        self.tile_w, self.tile_h = tile_size

        self._cache = {}

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        # Baca frame asli dari LazyVideo (hanya saat dibutuhkan)
        try:
            frame = self.video[idx]
        except Exception:
            return (
                np.zeros((self.rows * self.tile_h, self.cols * self.tile_w), dtype=np.uint8)
                if self.mode == "roi"
                else np.zeros((self.face_size[1], self.face_size[0], 3), dtype=np.uint8)
            )

        landmarks = self.landmarks[idx]

        if self.mode == "roi":
            # ROI Mode Assembly
            canvas = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), dtype=np.uint8)
            for j, roi_points in enumerate(self.roi_defs):
                roi, _ = self.landmarker.crop_roi(image=frame,
                                                  landmark_result=landmarks,
                                                  roi_points=roi_points,
                                                  margin=self.margin,
                                                  target_size=self.tile_size)
                r, c = divmod(j, self.cols)
                y1, y2 = r * self.tile_h, (r + 1) * self.tile_h
                x1, x2 = c * self.tile_w, (c + 1) * self.tile_w
                canvas[y1:y2, x1:x2] = roi
            res = canvas
        else:
            # Fullface Mode
            res = self.landmarker.crop(image=frame,
                                       landmarks=landmarks,
                                       margin=self.margin,
                                       output_size=self.face_size)

        self._cache[idx] = res
        return res


class _LazyFlows:
    """Helper class untuk hitung optical flow on-demand.

    T2-A: Memanfaatkan flow_cache dari streaming pass agar tidak menghitung
    ulang TVL1 yang sudah dihitung selama deteksi magnitude.
    """
    def __init__(self, lazy_frames: _LazyFrames, tvl1: TVL1,
                 flow_cache: Optional[dict] = None):
        self.frames = lazy_frames
        self.tvl1 = tvl1
        # T2-A: cache dari streaming pass (opsional) — hindari komputasi ulang
        self._cache = flow_cache if flow_cache is not None else {}

    def __len__(self):
        return len(self.frames) - 1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if idx in self._cache:
            return self._cache[idx]

        # Hitung flow jika tidak ada di cache
        prev = self.frames[idx]
        curr = self.frames[idx + 1]

        flow = self.tvl1.compute(prev, curr, download=True)
        self._cache[idx] = flow
        return flow


# T2-C: Ukuran batch untuk pengiriman TVL1 ke GPU
_FLOW_BATCH_SIZE = 16


class ApexPhaseSpotter:

    def __init__(self,
                 mode: ExtractionMode = "roi",
                 distance_threshold: int = 11,
                 prominence_threshold: float = 0.1,
                 tile_size: Tuple[int, int] = (64, 64),
                 face_size: Tuple[int, int] = (240, 240),
                 margin: float = 0.05,
                 prefetch_size: int = 8,
                 detect_interval: int = 3):

        self.mode = mode
        self.margin = margin
        self.prefetch_size = prefetch_size
        self.detect_interval = max(1, detect_interval)

        # Lazy init to support multiprocessing
        self.landmarker = None
        self.tvl1 = None

        self.apex_phase = ApexPhase(distance_threshold=distance_threshold,
                                    prominence_threshold=prominence_threshold)

        self.tile_w, self.tile_h = tile_size
        self.roi_defs = [frozenset(FaceRoiPoints.LEFT_EYE_POINTS),
                         frozenset(FaceRoiPoints.RIGHT_EYE_POINTS),
                         frozenset(FaceRoiPoints.LIPS_POINTS),
                         frozenset(FaceRoiPoints.LEFT_EYEBROW_POINTS),
                         frozenset(FaceRoiPoints.RIGHT_EYEBROW_POINTS)]

        self.cols = 3
        self.rows = math.ceil(len(self.roi_defs) / self.cols)
        self.face_size = face_size

        self.magnitudes: List[float] = []

        # Akan diisi dengan Lazy object, bukan list penuh
        self.flows = []
        self.frames = []


    def _lazy_init(self):
        """Initialize heavy objects here to ensure they are created in the worker process."""
        if self.landmarker is None:
            self.landmarker = FaceLandmark()
        if self.tvl1 is None:
            self.tvl1 = TVL1(fast_mode=True)


    def process(self, video_path: str) -> Tuple[List[int], List[int]]:
        self._lazy_init()
        self.magnitudes.clear()

        # Gunakan LazyVideo agar tidak load semua frame ke RAM
        video = LazyVideo(video_path)

        if len(video) < 2:
            return [], {}

        # Pass 1: Streaming Process (Hitung Magnitudes & Detect All Landmarks)
        # T2-A: streaming pass sekarang juga menyimpan flow ke _flow_cache
        all_landmarks, flow_cache = self._pipeline_process_streaming(video)

        # Pass 2: Setup Lazy Objects — _LazyFlows menggunakan flow_cache dari Pass 1
        self.frames = _LazyFrames(video, all_landmarks, self.mode,
                                  self.landmarker, self.roi_defs, self.margin,
                                  (self.tile_w, self.tile_h), self.face_size)

        # T2-A: teruskan flow_cache sehingga _LazyFlows tidak menghitung ulang
        self.flows = _LazyFlows(self.frames, self.tvl1, flow_cache=flow_cache)

        # Detect phases from magnitudes
        return self.__find_apex_phase(self.magnitudes)


    def _detect_all_landmarks(self, video: LazyVideo) -> List:
        """Deteksi landmark dengan interpolasi.

        T2-B: Menggunakan satu pass sequential decode (bukan random seek per keyframe)
        untuk menghindari biaya seek pada file AVI panjang.
        """
        n = len(video)
        interval = self.detect_interval

        keyframe_indices = list(range(0, n, interval))
        if keyframe_indices[-1] != n - 1:
            keyframe_indices.append(n - 1)

        keyframe_set = set(keyframe_indices)
        keyframe_landmarks = {}

        # T2-B: Sequential decode — buka capture baru dan baca frame satu per satu.
        # Jauh lebih cepat dari random seek (cap.set + cap.read) pada AVI panjang.
        cap = cv2.VideoCapture(video.video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in keyframe_set:
                keyframe_landmarks[frame_idx] = self.landmarker.detect(frame)
            frame_idx += 1
        cap.release()

        # Interpolasi landmark untuk frame di antara keyframe
        all_landmarks = [None] * n
        for idx in keyframe_indices:
            all_landmarks[idx] = keyframe_landmarks[idx]

        for k in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[k]
            end_idx = keyframe_indices[k + 1]
            span = end_idx - start_idx

            lm_start = keyframe_landmarks[start_idx]
            lm_end = keyframe_landmarks[end_idx]

            for i in range(start_idx + 1, end_idx):
                t = (i - start_idx) / span
                all_landmarks[i] = FaceLandmark.interpolate_landmarks(lm_start, lm_end, t)

        return all_landmarks


    def _pipeline_process_streaming(self, video: LazyVideo) -> Tuple[List, dict]:
        """Streaming pass: hitung magnitude dan deteksi landmark.

        T2-A: Flow disimpan ke flow_cache (dict idx → np.ndarray) sehingga
              _LazyFlows tidak perlu menghitung ulang TVL1 saat training.
        T2-B: _detect_all_landmarks kini menggunakan sequential decode.
        T2-C: TVL1 dikirim ke GPU dalam batch _FLOW_BATCH_SIZE pasang frame
              untuk meningkatkan GPU occupancy dan mengurangi Python call overhead.

        Returns:
            (all_landmarks, flow_cache)
        """
        # 1. T2-B: Detect landmarks via sequential decode
        all_landmarks = self._detect_all_landmarks(video)

        # 2. Sequential pass for flow magnitude + T2-A: populate flow_cache
        cap = cv2.VideoCapture(video.video_path)

        flow_cache: dict = {}
        prev_prep = None
        pending_pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []
        # pending_pairs: list of (prev_prep, curr_prep, flow_index)

        def _flush_batch():
            """Kirim batch frame pairs ke TVL1 dan simpan hasilnya."""
            if not pending_pairs:
                return
            frame_pairs = [(p, c) for p, c, _ in pending_pairs]

            # T2-C: compute_batch kirim semua pairs sekaligus ke GPU
            flows = self.tvl1.compute_batch(frame_pairs, download=True)

            for (_, _, flow_idx), flow in zip(pending_pairs, flows):
                flow_cache[flow_idx] = flow
                mag = float(np.mean(np.hypot(flow[..., 0], flow[..., 1])))
                self.magnitudes.append(mag)

            pending_pairs.clear()

        frame_count = len(video)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            curr_landmarks = all_landmarks[i]

            # Preprocess (Crop) based on mode
            if self.mode == "roi":
                curr_prep = self._assemble_roi(frame, curr_landmarks)
            else:
                curr_prep = self.landmarker.crop(frame, curr_landmarks, self.margin, self.face_size)

            if i > 0 and prev_prep is not None:
                pending_pairs.append((prev_prep, curr_prep, i - 1))

                # T2-C: tuang ke GPU setiap _FLOW_BATCH_SIZE pasang
                if len(pending_pairs) == _FLOW_BATCH_SIZE:
                    _flush_batch()

            prev_prep = curr_prep

        # Tuang sisa pasang yang belum diproses
        _flush_batch()
        cap.release()

        return all_landmarks, flow_cache


    def _assemble_roi(self, frame, landmarks):
        """Helper to assemble ROI canvas during streaming pass."""
        canvas = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), dtype=np.uint8)
        for j, roi_points in enumerate(self.roi_defs):
            roi, _ = self.landmarker.crop_roi(frame, landmarks, roi_points, self.margin, (self.tile_w, self.tile_h))
            r, c = divmod(j, self.cols)
            y1, y2 = r * self.tile_h, (r + 1) * self.tile_h
            x1, x2 = c * self.tile_w, (c + 1) * self.tile_w
            canvas[y1:y2, x1:x2] = roi
        return canvas


    def __find_apex_phase(self, magnitudes: List[float]) -> Tuple[List[int], List[int]]:
        if not magnitudes:
            return [], {}

        window_length = ApexSmoother.calculate_window_length(len(magnitudes))
        polyorder = ApexSmoother.calculate_polyorder(window_length)
        smoothed = savgol_filter(magnitudes, window_length, polyorder)

        apex_indices = self.apex_phase.find_apex(signal=smoothed)
        phases = self.apex_phase.find_phase(signal=smoothed, apex_indices=apex_indices)
        return apex_indices, phases