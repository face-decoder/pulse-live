import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

from pathlib import Path
from typing import List, Tuple

import absl.logging
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

absl.logging.set_verbosity(absl.logging.ERROR)


class FaceLandmark:
    BASE_ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent

    BASE_MODEL_PATH: str = "src/face/tasks/face_landmarker.task"

    MODEL_PATH: Path = BASE_ROOT_DIR / BASE_MODEL_PATH

    base_options: python.BaseOptions  # type: ignore

    options: vision.FaceLandmarkerOptions  # type: ignore

    landmarker: vision.FaceLandmarker  # type: ignore

    landmark: vision.FaceLandmarkerResult | None  # type: ignore

    FACE_OVAL: List[int] = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]

    STABLE_POINTS: List[int] = [
        1,
        33,
        263,
        61,
        291,
    ]

    def __init__(self):

        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")

        self.base_options = python.BaseOptions(
            model_asset_path=self.MODEL_PATH.as_posix()
        )

        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            num_faces=1,
            min_tracking_confidence=0.7,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            running_mode=vision.RunningMode.IMAGE,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(self.options)

        self.landmark = None

    @staticmethod
    def interpolate_landmarks(landmarks_a, landmarks_b, t: float):
        if not landmarks_a.face_landmarks or not landmarks_b.face_landmarks:
            return landmarks_a if landmarks_a.face_landmarks else landmarks_b

        lm_a = landmarks_a.face_landmarks[0]
        lm_b = landmarks_b.face_landmarks[0]

        interpolated_points = []
        for pa, pb in zip(lm_a, lm_b):
            interpolated_points.append(
                _InterpolatedLandmark(
                    x=pa.x + (pb.x - pa.x) * t,
                    y=pa.y + (pb.y - pa.y) * t,
                    z=pa.z + (pb.z - pa.z) * t,
                )
            )

        return _InterpolatedResult(interpolated_points)

    def detect(self, image: np.ndarray) -> vision.FaceLandmarkerResult:  # type: ignore

        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray.")

        if image.size == 0:
            raise ValueError("Input image is empty.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        self.landmark = self.landmarker.detect(mp_image)
        return self.landmark

    def crop(
        self,
        image: np.ndarray,
        landmarks: vision.FaceLandmarkerResult = None,  # type: ignore
        landmark_indices: List[int] = None,
        margin: float = 0.05,
        output_size: Tuple[int, int] = (240, 240),
    ) -> np.ndarray:

        if landmarks is None and self.landmark is None:
            raise ValueError("Landmark detection has not been performed.")

        final_landmarks = landmarks if landmarks is not None else self.landmark

        if not final_landmarks.face_landmarks:
            raise ValueError("No face landmarks detected.")

        if landmark_indices is None:
            landmark_indices = self.FACE_OVAL

        effective_indices = landmark_indices + self.STABLE_POINTS

        h, w, _ = image.shape
        face_landmarks = final_landmarks.face_landmarks[0]

        xs = [face_landmarks[i].x * w for i in effective_indices]
        ys = [face_landmarks[i].y * h for i in effective_indices]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        dx = int((x_max - x_min) * margin)
        dy = int((y_max - y_min) * margin)

        x_min = max(0, x_min - dx)
        x_max = min(w, x_max + dx)
        y_min = max(0, y_min - dy)
        y_max = min(h, y_max + dy)

        face_crop = image[y_min:y_max, x_min:x_max]
        face_crop = cv2.resize(face_crop, output_size)

        return face_crop

    def crop_roi(
        self,
        image: np.ndarray,
        landmark_result: vision.FaceLandmarkerResult,  # type: ignore
        roi_points: frozenset,
        margin: float = 0.05,
        target_size: Tuple[int, int] = (64, 64),
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray.")

        if image.size == 0:
            raise ValueError("Input image is empty.")

        if landmark_result is None or not landmark_result.face_landmarks:
            raise ValueError("No face landmarks detected.")

        h, w = image.shape[:2]

        landmarks = landmark_result.face_landmarks[0]

        roi_indices = set()
        for a, b in roi_points:
            roi_indices.add(a)
            roi_indices.add(b)

        xs = [landmarks[i].x * w for i in roi_indices]
        ys = [landmarks[i].y * h for i in roi_indices]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        dx = (x_max - x_min) * margin
        dy = (y_max - y_min) * margin

        x1 = int(max(0, x_min - dx))
        y1 = int(max(0, y_min - dy))
        x2 = int(min(w, x_max + dx))
        y2 = int(min(h, y_max + dy))

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            raise ValueError("Empty ROI after cropping.")

        th, tw = target_size
        rh, rw = roi.shape[:2]

        scale = min(tw / rw, th / rh)

        new_w = int(rw * scale)
        new_h = int(rh * scale)

        resized = cv2.resize(roi, (new_w, new_h))

        output = np.zeros((th, tw, 3), dtype=roi.dtype)
        x_off = (tw - new_w) // 2
        y_off = (th - new_h) // 2

        output[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        mask = np.zeros((th, tw), dtype=np.uint8)

        roi_polygon = []
        for i in roi_indices:
            px = landmarks[i].x * w - x1
            py = landmarks[i].y * h - y1

            px = px * scale + x_off
            py = py * scale + y_off

            roi_polygon.append([int(px), int(py)])

        roi_polygon = np.array(roi_polygon, dtype=np.int32)
        cv2.fillConvexPoly(mask, roi_polygon, 255)

        return output, mask


class _InterpolatedLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class _InterpolatedResult:
    __slots__ = ("face_landmarks",)

    def __init__(self, landmarks: list):
        self.face_landmarks = [landmarks]
