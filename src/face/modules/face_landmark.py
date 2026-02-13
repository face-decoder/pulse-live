import os
import cv2
import numpy as np
import absl.logging
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, List


# Mencoba untuk supress (menyembunyikan) log warning dari TensorFlow mediapipe
# Harapan: untuk membuat output lebih bersih tanpa log yang tidak berpengaruh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
absl.logging.set_verbosity(absl.logging.ERROR)


class FaceLandmark:

    BASE_ROOT_DIR: str = Path(os.getcwd()).parent

    BASE_MODEL_PATH: str = "src/face/tasks/face_landmarker.task"

    MODEL_PATH: Path = Path(BASE_ROOT_DIR, BASE_MODEL_PATH)

    base_options: python.BaseOptions #type: ignore

    options: vision.FaceLandmarkerOptions #type: ignore

    landmarker: vision.FaceLandmarker #type: ignore

    landmark: vision.FaceLandmarkerResult | None #type: ignore

    FACE_OVAL: List[int] = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]

    STABLE_POINTS: List[int] = [
        1,    # nose tip
        33,   # left eye outer
        263,  # right eye outer
        61,   # mouth left
        291   # mouth right
    ]


    def __init__(self):

        # Jika file model tidak ditemukan, raise dengan sebuah error
        # untuk menghindari kesalahan saat inisialisasi model FaceLandmark
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")

        # Inisialisasi base options dengan menyertakan path model yang benar
        self.base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH.as_posix())

        # Inisialisasi options untuk FaceLandmarker
        # Beberapa parameter ditambahkan untuk konfigurasi yang lebih baik, seperti:
        # num_faces -> untuk menentukan jumlah wajah yang akan dideteksi
        # min_tracking_confidence -> untuk mengatur ambang kepercayaan pelacakan wajah
        # min_face_detection_confidence -> untuk mengatur ambang kepercayaan deteksi wajah
        # min_face_presence_confidence -> untuk mengatur ambang kepercayaan keberadaan wajah
        # running_mode -> untuk menentukan mode operasi (IMAGE, VIDEO, LIVE_STREAM)
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                                    num_faces=1,
                                                    min_tracking_confidence=0.7,
                                                    min_face_detection_confidence=0.7,
                                                    min_face_presence_confidence=0.7,
                                                    running_mode=vision.RunningMode.IMAGE)
        
        # Membuat instance Face Landmarker dengan opsi yang telah ditentukan
        self.landmarker = vision.FaceLandmarker.create_from_options(self.options)

        # Membuat attribut landmark untuk menyimpan hasil deteksi
        self.landmark = None


    def detect(self, image: np.ndarray) -> vision.FaceLandmarkerResult: #type: ignore
        """
        Melakukan deteksi landmarks dengan citra input

        Args:
            image: Citra input dalam format numpy ndarray

        Returns:
            FaceLandmarkerResult: Hasil deteksi dari FaceLandmarker

        Raises:
            ValueError: Jika citra input tidak valid
        """

        # Jika citra bukan merupakan instance dari numpy ndarray
        # Maka raise dengan ValueError yang menunjukkan image harus berupa numpy ndarray
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray.")

        # Jika citra tidak memiliki value apapun meskipun instance dari numpy ndarray
        # Maka raise dengan ValueError yang menunjukkan image tidak boleh kosong (invalid)
        if image.size == 0:
            raise ValueError("Input image is empty.")

        # Memastikan citra dalam format RGB sebelum diproses
        # Hal ini diperlukan oleh MediaPipe Face Landmarker
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Melakukan deteksi landmarks pada citra RGB
        self.landmark = self.landmarker.detect(mp_image)
        return self.landmark


    def crop(self,
             image: np.ndarray,
             landmarks: vision.FaceLandmarkerResult = None,  # type: ignore
             landmark_indices: List[int] = None,
             margin: float = 0.05,
             output_size: Tuple[int, int] = (240, 240)) -> np.ndarray:
        """
        Melakukan cropping wajah berdasarkan landmark yang terdeteksi

        Args:
            image: Citra input dalam format numpy ndarray
            landmarks: Hasil deteksi landmark (opsional)
            landmark_indices: Indeks landmark yang akan digunakan untuk cropping (opsional)
            margin: Margin tambahan untuk bounding box (default: 0.05)
            output_size: Ukuran output citra setelah cropping dan resizing (default: (240, 240))

        Returns:
            np.ndarray: Citra wajah yang telah di-crop dan di-resize

        Raises:
            ValueError: Jika landmark tidak tersedia atau tidak ada landmark yang terdeteksi
        """

        # Validasi landmark yang diberikan telah sesuai atau tidak
        # Jika tidak ada landmark yang diberikan dan juga tidak ada landmark yang tersimpan
        if landmarks is None and self.landmark is None:
            raise ValueError("Landmark detection has not been performed.")

        # Gunakan landmark yang diberikan atau yang tersimpan
        final_landmarks = landmarks if landmarks is not None else self.landmark

        # Jika tidak ada landmark yang terdeteksi
        # Maka raise dengan ValueError yang menunjukkan tidak ada landmark yang terdeteksi
        if not final_landmarks.face_landmarks:
            raise ValueError("No face landmarks detected.")

        # Gunakan indeks landmark default jika tidak diberikan
        if landmark_indices is None:
            landmark_indices = self.FACE_OVAL

        # Gabungkan dengan titik stabil untuk memastikan cropping yang konsisten
        effective_indices = landmark_indices + self.STABLE_POINTS

        h, w, _ = image.shape
        face_landmarks = final_landmarks.face_landmarks[0]

        # Ambil koordinat landmark subset yang diinginkan
        xs = [face_landmarks[i].x * w for i in effective_indices]
        ys = [face_landmarks[i].y * h for i in effective_indices]

        # Menghitung bounding box dari landmark yang diberikan
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        # Menghitung margin tambahan untuk bounding box
        dx = int((x_max - x_min) * margin)
        dy = int((y_max - y_min) * margin)

        x_min = max(0, x_min - dx)
        x_max = min(w, x_max + dx)
        y_min = max(0, y_min - dy)
        y_max = min(h, y_max + dy)

        # Memotong citra berdasarkan bounding box yang dihitung
        face_crop = image[y_min:y_max, x_min:x_max]
        face_crop = cv2.resize(face_crop, output_size)

        return face_crop


    def crop_roi(self,
                 image: np.ndarray, 
                 landmark_result: vision.FaceLandmarkerResult, #type: ignore
                 roi_points: frozenset,
                 margin: float = 0.05,
                 target_size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memotong bagian wajah berdasarkan region of interests (RoI) yang telah didefinisikan

        Args:
            image (np.ndarray): Citra input dalam format numpy ndarray
            landmark_result (FaceLandmarkerResult): Hasil deteksi landmark dari FaceLandmarker
            roi_points (frozenset): Koneksi titik landmark yang mendefinisikan RoI
            margin (float, optional): Margin tambahan untuk bounding box. Default adalah 0.05
            target_size (Tuple[int, int], optional): Ukuran target output citra RoI. Default adalah (64, 64)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple berisi citra RoI yang telah dipotong dan mask-nya
            
        Raises:
            ValueError: Jika citra input atau hasil deteksi landmark tidak valid
        """
        
        # Validasi citra input berupa numpy ndarray
        # Jika tidak, maka raise dengan ValueError
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray.")
        
        # Validasi citra input tidak kosong
        # Jika kosong, maka raise dengan ValueError
        if image.size == 0:
            raise ValueError("Input image is empty.")
        
        # Validasi hasil deteksi landmark
        # Jika tidak ada hasil deteksi, maka raise dengan ValueError
        if landmark_result is None or not landmark_result.face_landmarks:
            raise ValueError("No face landmarks detected.")
        
        h, w = image.shape[:2]

        landmarks = landmark_result.face_landmarks[0]

        # Mengumpulkan semua indeks landmark yang terlibat dalam RoI
        roi_indices = set()
        for a, b in roi_points:
            roi_indices.add(a)
            roi_indices.add(b)

        # Menghitung bounding box dari landmark RoI
        xs = [landmarks[i].x * w for i in roi_indices]
        ys = [landmarks[i].y * h for i in roi_indices]

        # Menghitung koordinat bounding box untuk margin tambahan
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Menghitung tambahan margin tambahan untuk bounding box
        dx = (x_max - x_min) * margin
        dy = (y_max - y_min) * margin

        # Menghitung koordinat akhir bounding box dengan margin
        x1 = int(max(0, x_min - dx))
        y1 = int(max(0, y_min - dy))
        x2 = int(min(w, x_max + dx))
        y2 = int(min(h, y_max + dy))

        # Memotong citra berdasarkan bounding box yang dihitung
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            raise ValueError("Empty ROI after cropping.")

        th, tw = target_size
        rh, rw = roi.shape[:2]

        # Menghitung skala untuk resizing sambil mempertahankan aspek rasio
        scale = min(tw / rw, th / rh)

        # Menghitung ukuran baru setelah scaling
        new_w = int(rw * scale)
        new_h = int(rh * scale)

        # Melakukan resizing pada RoI
        resized = cv2.resize(roi, (new_w, new_h))

        # Membuat citra output dengan ukuran target dan mengisi dengan nol (hitam)
        output = np.zeros((th, tw, 3), dtype=roi.dtype)
        x_off = (tw - new_w) // 2
        y_off = (th - new_h) // 2

        # Menempatkan citra yang telah di-resize ke dalam citra output di tengah
        output[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Menghitung masking area
        mask = np.zeros((th, tw), dtype=np.uint8)

        # Menghitung polygon RoI pada citra output
        roi_polygon = []
        for i in roi_indices:
            px = landmarks[i].x * w - x1
            py = landmarks[i].y * h - y1

            px = px * scale + x_off
            py = py * scale + y_off

            roi_polygon.append([int(px), int(py)])

        # Mengisi polygon pada mask
        roi_polygon = np.array(roi_polygon, dtype=np.int32)
        cv2.fillConvexPoly(mask, roi_polygon, 255)

        return output, mask