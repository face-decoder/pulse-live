import cv2
import numpy as np

from src.video.modules import Video
from src.face.modules import FaceLandmark, FaceRoiPoints, FaceRoiSizes
from src.optical_flow.modules import TVL1
from src.apex.modules.v2 import ApexPhase, ApexSmoother

from typing import Dict, List, Literal, Tuple, Optional


class ApexPhaseSpotter:

    def __init__(
        self,
        margin: float = 0.05,
        mode: Literal["single", "batch"] = "single"
    ):
        self.margin = margin
        self.mode = mode

        self.face_landmark = FaceLandmark()
        self.tvl1 = TVL1()
        self.apex_phase = ApexPhase()
        self.smoother = ApexSmoother()

        self.roi_defs: List[Tuple[str, object, Tuple[int, int]]] = [
            ("left_eye", FaceRoiPoints.LEFT_EYE_POINTS, FaceRoiSizes.EYE_SIZE),
            ("right_eye", FaceRoiPoints.RIGHT_EYE_POINTS, FaceRoiSizes.EYE_SIZE),
            ("lips", FaceRoiPoints.LIPS_POINTS, FaceRoiSizes.LIPS_SIZE),
            ("left_eyebrow", FaceRoiPoints.LEFT_EYEBROW_POINTS, FaceRoiSizes.EYEBROW_SIZE),
            ("right_eyebrow", FaceRoiPoints.RIGHT_EYEBROW_POINTS, FaceRoiSizes.EYEBROW_SIZE),
        ]

        self._validate_roi_defs()
        self._reset()


    def _validate_roi_defs(self) -> None:
        for i, item in enumerate(self.roi_defs):
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(
                    f"roi_defs[{i}] must be (roi_name, roi_points, roi_size), got: {item}"
                )

    def close(self) -> None:
        if self.tvl1 is not None and hasattr(self.tvl1, "close"):
            self.tvl1.close()


    def _reset(self) -> None:

        self.magnitudes: List[float] = []

        self.horizontal_magnitudes: Dict[str, List[np.ndarray]] = {
            roi_name: [] for roi_name, _, _ in self.roi_defs
        }
        
        self.vertical_magnitudes: Dict[str, List[np.ndarray]] = {
            roi_name: [] for roi_name, _, _ in self.roi_defs
        }

        self.frame_roi_flows: List[List[Dict[str, np.ndarray]]] = []

        self._cached_landmarks = None


    def _append_roi_flow(self,
                         roi_name: str,
                         dx: np.ndarray,
                         dy: np.ndarray,
                         frame_bucket: Optional[List[Dict[str, np.ndarray]]] = None) -> None:
                         
        dx_arr = np.asarray(dx, dtype=np.float32)
        
        dy_arr = np.asarray(dy, dtype=np.float32)

        self.horizontal_magnitudes[roi_name].append(dx_arr)

        self.vertical_magnitudes[roi_name].append(dy_arr)

        if frame_bucket is not None:
            frame_bucket.append({ "roi": roi_name,
                                  "dx": dx_arr,
                                  "dy": dy_arr })


    def export_flow_data(self) -> dict:
        frame_count = len(self.magnitudes)
        landmark_detected_count = sum(1 for frame in self.frame_roi_flows if len(frame) > 0)
        landmark_detection_rate = landmark_detected_count / frame_count if frame_count > 0 else 0.0
        roi_order = [roi[0] for roi in self.roi_defs]

        return {
            "frames": self.frame_roi_flows,
            "horizontal_magnitudes": {
                roi_name: (
                    np.stack(values, axis=0)
                    if len(values) > 0 else np.empty((0,), dtype=np.float32)
                )
                for roi_name, values in self.horizontal_magnitudes.items()
            },
            "vertical_magnitudes": {
                roi_name: (
                    np.stack(values, axis=0)
                    if len(values) > 0 else np.empty((0,), dtype=np.float32)
                )
                for roi_name, values in self.vertical_magnitudes.items()
            },
            "magnitudes": np.asarray(self.magnitudes, dtype=np.float32),
            "frame_count": frame_count,
            "landmark_detection_rate": landmark_detection_rate,
            "roi_order": roi_order,
        }


    def process(self, video_path: str):
        """
        Proses satu video dan kembalikan apex indices beserta phases.
        
        Args:
            video_path: Path ke file video yang akan diproses.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada video.
        """

        self._reset()

        video = Video(video_path=video_path)
        video.map(self.__process_roi)

        return self.__find_apex_phase(self.magnitudes)


    def process_videos(self, video_paths: List[str]):
        """
        Proses beberapa video secara berurutan dan kembalikan apex indices beserta phases.
        
        Args:
            video_paths: List of paths ke file video yang akan diproses.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada semua video.
        """

        self._reset()

        for path in video_paths:
            video = Video(video_path=path)
            video.map(self.__process_roi)

        return self.__find_apex_phase(self.magnitudes)
    

    def process_image_list(self, image_list: List[str]):
        """
        Proses list of images yang sudah diurutkan sebagai frame video.

        Args:
            image_list: List of paths ke file gambar yang akan diproses sebagai frame video.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada sequence gambar.
        """

        self._reset()

        for i in range(len(image_list) - 1):
            prev_frame = cv2.imread(image_list[i])
            curr_frame = cv2.imread(image_list[i + 1])
            self.__process_roi(prev_frame, curr_frame, i)

        return self.__find_apex_phase(self.magnitudes)


    def process_frames(self, frames: List[np.ndarray]):
        """
        Proses list of in-memory numpy frames.

        Args:
            frames: List of numpy arrays representing video frames.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada sequence gambar.
        """
        
        self._reset()

        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            curr_frame = frames[i + 1]
            self.__process_roi(prev_frame, curr_frame, i)

        return self.__find_apex_phase(self.magnitudes)


    def __detect_landmarks(self, prev_frame: np.ndarray, curr_frame: np.ndarray):
        """
        Deteksi landmark dengan memanfaatkan cache.
        Setiap frame hanya dideteksi satu kali meskipun muncul sebagai
        prev_frame pada iterasi berikutnya.
        """
        prev_landmarks = (
            self._cached_landmarks
            if self._cached_landmarks is not None
            else self.face_landmark.detect(prev_frame)
        )

        curr_landmarks = self.face_landmark.detect(curr_frame)
        
        self._cached_landmarks = curr_landmarks

        return prev_landmarks, curr_landmarks


    def __extract_rois(self,
                       prev_frame: np.ndarray,
                       curr_frame: np.ndarray,
                       prev_landmarks,
                       curr_landmarks):
        roi_items = []

        for roi_name, roi_points, roi_size in self.roi_defs:
            roi_prev, mask_prev = self.face_landmark.crop_roi(
                image=prev_frame,
                landmark_result=prev_landmarks,
                roi_points=roi_points,
                margin=self.margin,
                target_size=roi_size
            )

            roi_next, mask_next = self.face_landmark.crop_roi(
                image=curr_frame,
                landmark_result=curr_landmarks,
                roi_points=roi_points,
                margin=self.margin,
                target_size=roi_size
            )

            if roi_prev is None or roi_next is None:
                continue

            roi_items.append((roi_name, roi_prev, roi_next, mask_prev, mask_next))

        return roi_items


    def __process_roi(self, prev_frame: np.ndarray, curr_frame: np.ndarray, frame_index: int):
        prev_landmarks, curr_landmarks = self.__detect_landmarks(prev_frame, curr_frame)

        roi_items = self.__extract_rois(
            prev_frame,
            curr_frame,
            prev_landmarks,
            curr_landmarks
        )

        frame_bucket: List[Dict[str, np.ndarray]] = []

        if not roi_items:
            self.magnitudes.append(0.0)
            self.frame_roi_flows.append(frame_bucket)  # frame kosong
            return

        if self.mode == "batch":
            frame_magnitude = self.__compute_batch(roi_items, frame_bucket)
        else:
            frame_magnitude = self.__compute_single(roi_items, frame_bucket)

        self.magnitudes.append(frame_magnitude)
        self.frame_roi_flows.append(frame_bucket)


    def __compute_single(self, roi_items: list, frame_bucket: List[Dict[str, np.ndarray]]):
        roi_magnitudes = []

        for roi_name, roi_prev, roi_next, mask_prev, mask_next in roi_items:
            flow = self.tvl1.compute(roi_prev, roi_next)

            dx = np.asarray(flow[..., 0], dtype=np.float32)
            dy = np.asarray(flow[..., 1], dtype=np.float32)

            self._append_roi_flow(roi_name, dx, dy, frame_bucket)

            magnitude = np.hypot(dx, dy)
            valid = (mask_prev > 0) & (mask_next > 0)

            if np.any(valid):
                roi_mean = float(np.mean(magnitude[valid]))
            else:
                roi_mean = float(np.mean(magnitude))

            roi_magnitudes.append(roi_mean)

        return float(np.mean(roi_magnitudes)) if roi_magnitudes else 0.0


    def __compute_batch(self, roi_items: list, frame_bucket: List[Dict[str, np.ndarray]]):
        max_w = max(roi_prev.shape[1] for _, roi_prev, _, _, _ in roi_items)

        padded_items = []

        for roi_name, roi_prev, roi_next, mask_prev, mask_next in roi_items:
            pad_w = max_w - roi_prev.shape[1]

            if pad_w > 0:
                roi_prev = np.pad(roi_prev, ((0, 0), (0, pad_w), (0, 0)), mode="constant")
                roi_next = np.pad(roi_next, ((0, 0), (0, pad_w), (0, 0)), mode="constant")
                mask_prev = np.pad(mask_prev, ((0, 0), (0, pad_w)), mode="constant")
                mask_next = np.pad(mask_next, ((0, 0), (0, pad_w)), mode="constant")

            padded_items.append((roi_name, roi_prev, roi_next, mask_prev, mask_next))

        batch_prev = np.vstack([x[1] for x in padded_items])
        batch_next = np.vstack([x[2] for x in padded_items])

        flow = self.tvl1.compute(batch_prev, batch_next)
        dx_batch = np.asarray(flow[..., 0], dtype=np.float32)
        dy_batch = np.asarray(flow[..., 1], dtype=np.float32)
        magnitude_batch = np.hypot(dx_batch, dy_batch)

        roi_magnitudes = []
        y_offset = 0

        for roi_name, roi_prev, _, mask_prev, mask_next in padded_items:
            h = roi_prev.shape[0]

            dx = dx_batch[y_offset:y_offset + h, :]
            dy = dy_batch[y_offset:y_offset + h, :]
            magnitude = magnitude_batch[y_offset:y_offset + h, :]

            self._append_roi_flow(roi_name, dx, dy, frame_bucket)

            valid = (mask_prev > 0) & (mask_next > 0)
            if np.any(valid):
                roi_mean = float(np.mean(magnitude[valid]))
            else:
                roi_mean = float(np.mean(magnitude))

            roi_magnitudes.append(roi_mean)
            y_offset += h

        return float(np.mean(roi_magnitudes)) if roi_magnitudes else 0.0


    def __find_apex_phase(self, magnitudes: List[float]):

        smoothed = ApexSmoother.smooth(signal=magnitudes)

        wl = ApexSmoother.calculate_window_length(len(magnitudes))
        po = ApexSmoother.calculate_polyorder(wl)

        signal_arr = np.array(smoothed)
        height_threshold = float(np.mean(signal_arr) + np.std(signal_arr))

        self.smoothed_magnitudes = smoothed

        apex_indices = self.apex_phase.find_top_k_apex(smoothed, k=10, height=height_threshold)

        phases = self.apex_phase.find_phase(signal=smoothed,
                                            apex_indices=apex_indices,
                                            cutoff_ratio=0.35)

        return apex_indices, phases