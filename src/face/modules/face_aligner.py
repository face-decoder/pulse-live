import logging

import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

logger = logging.getLogger(__name__)


class FaceAlignerPoints:
    # Titik mata kiri (beberapa titik untuk rata-rata yang lebih akurat)
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    # Titik mata kanan
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Titik dagu & dahi (untuk estimasi tinggi wajah)
    CHIN = 152
    FOREHEAD = 10

    # Titik hidung dan mulut
    NOSE_TIP = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291


class FaceAligner:
    def __init__(
        self,
        output_size: tuple[int, int] = None,
        eye_center_ratio: float = 0.28,
        eye_width_ratio: float = 0.38,
        debug: bool = False,
    ):
        """
        Args:
            output_size     : Ukuran output (width, height). Jika None, sama dengan input.
            eye_center_ratio: Posisi vertikal pusat mata relatif terhadap tinggi output.
                              0.28 = mata berada 28% dari atas, memberi ruang lebih
                              untuk dagu dan bibir di bawah. Default: 0.28.
            eye_width_ratio : Jarak antar mata relatif terhadap lebar output.
                              Nilai lebih kecil = wajah lebih kecil / lebih banyak konteks.
                              Default: 0.38 (zoom lebih keluar agar dagu tidak terpotong).
            debug           : Jika True, log nilai landmark dan transform matrix.
        """
        self.output_size = output_size
        self.eye_center_ratio = eye_center_ratio
        self.eye_width_ratio = eye_width_ratio
        self.debug = debug

    # ------------------------------------------------------------------
    # Helper: rata-rata beberapa titik landmark → koordinat piksel (x, y)
    # ------------------------------------------------------------------
    def _eye_center(
        self, landmark, indices: list[int], img_w: int, img_h: int
    ) -> np.ndarray:
        """
        Mengembalikan pusat mata dalam koordinat piksel.

        MediaPipe menyimpan landmark sebagai (x=col/width, y=row/height),
        sehingga konversi yang benar adalah:
            pixel_x = landmark.x * image_width   (arah horizontal)
            pixel_y = landmark.y * image_height  (arah vertikal)
        """
        pts = np.array(
            [[lm.x * img_w, lm.y * img_h] for lm in (landmark[i] for i in indices)],
            dtype=np.float64,
        )
        return pts.mean(axis=0)  # shape: (2,) → [x, y]

    # ------------------------------------------------------------------
    # Helper: satu titik landmark → koordinat piksel
    # ------------------------------------------------------------------
    def _pixel(self, lm, img_w: int, img_h: int) -> np.ndarray:
        return np.array([lm.x * img_w, lm.y * img_h], dtype=np.float64)

    # ------------------------------------------------------------------
    # Align
    # ------------------------------------------------------------------
    def align(
        self,
        image: np.ndarray | None = None,
        landmarks: FaceLandmarkerResult | None = None,
    ) -> np.ndarray:
        """
        Menyelaraskan wajah menggunakan similarity transform murni
        (rotasi + skala seragam + translasi, tanpa shear).

        Pipeline:
          1. Rata-rata 4 titik per mata → pusat mata yang stabil
          2. Hitung sudut kemiringan dari garis antar-mata
          3. Hitung skala dari jarak antar-mata vs target
          4. Bangun matriks 2×3 similarity transform
          5. warpAffine ke kanvas output

        Args:
            image    : BGR/RGB numpy array.
            landmarks: FaceLandmarkerResult dari MediaPipe.

        Returns:
            numpy array ukuran (out_h, out_w, C) yang sudah diselaraskan.

        Raises:
            ValueError: Input tidak valid atau tidak ada wajah.
        """
        # --- Validasi input ---
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid numpy ndarray.")
        if not isinstance(landmarks, FaceLandmarkerResult):
            raise ValueError(
                "Input landmarks must be a valid FaceLandmarkerResult instance."
            )
        if not landmarks.face_landmarks:
            raise ValueError("No face detected in the image.")

        lm = landmarks.face_landmarks[0]

        # image.shape = (rows, cols, channels) = (height, width, C)
        img_h, img_w = image.shape[:2]
        out_w, out_h = self.output_size if self.output_size else (img_w, img_h)

        # ----------------------------------------------------------------
        # 1. Pusat mata dalam piksel
        #    Penting: img_w dikalikan dengan .x, img_h dikalikan dengan .y
        # ----------------------------------------------------------------
        left_eye = self._eye_center(
            lm,
            [
                FaceAlignerPoints.LEFT_EYE_INNER,
                FaceAlignerPoints.LEFT_EYE_OUTER,
                FaceAlignerPoints.LEFT_EYE_TOP,
                FaceAlignerPoints.LEFT_EYE_BOTTOM,
            ],
            img_w,
            img_h,
        )
        right_eye = self._eye_center(
            lm,
            [
                FaceAlignerPoints.RIGHT_EYE_INNER,
                FaceAlignerPoints.RIGHT_EYE_OUTER,
                FaceAlignerPoints.RIGHT_EYE_TOP,
                FaceAlignerPoints.RIGHT_EYE_BOTTOM,
            ],
            img_w,
            img_h,
        )

        # ----------------------------------------------------------------
        # 2. Sudut kemiringan
        #    delta = right_eye - left_eye  →  (Δx, Δy) dalam piksel
        #    arctan2(Δy, Δx) memberi sudut kemiringan garis antar-mata
        #    terhadap sumbu horizontal.  Nilai positif = mata kanan lebih
        #    rendah; nilai negatif = mata kanan lebih tinggi.
        # ----------------------------------------------------------------
        delta = right_eye - left_eye  # [Δx, Δy]
        angle_rad = np.arctan2(delta[1], delta[0])
        angle_deg = float(np.degrees(angle_rad))

        # ----------------------------------------------------------------
        # 3. Skala
        # ----------------------------------------------------------------
        eye_dist = float(np.linalg.norm(delta)) + 1e-8
        target_eye_dist = self.eye_width_ratio * out_w
        scale = target_eye_dist / eye_dist

        # ----------------------------------------------------------------
        # 4. Similarity transform matrix
        #
        #    Rotasi −angle agar garis mata menjadi horizontal, lalu skala,
        #    lalu translasi agar midpoint mata mendarat tepat di target.
        #
        #    M = [ cos  -sin  tx ]
        #        [ sin   cos  ty ]
        #
        #    di mana cos/sin sudah dikalikan scale.
        # ----------------------------------------------------------------
        ca = np.cos(-angle_rad) * scale  # rotasi berlawanan arah kemiringan
        sa = np.sin(-angle_rad) * scale

        midpoint_src = (left_eye + right_eye) / 2.0
        target_mid = np.array(
            [out_w * 0.5, out_h * self.eye_center_ratio], dtype=np.float64
        )

        # Translasi: setelah rotasi+skala diterapkan pada midpoint_src
        # hasilnya harus sama dengan target_mid
        tx = target_mid[0] - (ca * midpoint_src[0] - sa * midpoint_src[1])
        ty = target_mid[1] - (sa * midpoint_src[0] + ca * midpoint_src[1])

        M = np.array(
            [
                [ca, -sa, tx],
                [sa, ca, ty],
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------------------
        # Debug log — aktifkan dengan debug=True
        # ----------------------------------------------------------------
        if self.debug:
            chin = self._pixel(lm[FaceAlignerPoints.CHIN], img_w, img_h)
            forehead = self._pixel(lm[FaceAlignerPoints.FOREHEAD], img_w, img_h)
            face_h = float(np.linalg.norm(chin - forehead))
            logger.debug(
                "FaceAligner debug:\n"
                "  image shape      : %s  (h=%d, w=%d)\n"
                "  left_eye  (px)   : [%.1f, %.1f]\n"
                "  right_eye (px)   : [%.1f, %.1f]\n"
                "  eye_distance (px): %.1f\n"
                "  face_height (px) : %.1f\n"
                "  angle_deg        : %.2f°\n"
                "  scale            : %.4f\n"
                "  midpoint_src     : [%.1f, %.1f]\n"
                "  target_mid       : [%.1f, %.1f]\n"
                "  transform M      :\n%s",
                image.shape,
                img_h,
                img_w,
                *left_eye,
                *right_eye,
                eye_dist,
                face_h,
                angle_deg,
                scale,
                *midpoint_src,
                *target_mid,
                np.array2string(M, precision=4),
            )

        # ----------------------------------------------------------------
        # 5. Terapkan transformasi
        # ----------------------------------------------------------------
        aligned = cv2.warpAffine(
            image,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned


    def align_with_landmarks(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarkerResult,
    ) -> tuple:
        """
        Sama seperti align(), tetapi mengembalikan (aligned_image, transformed_landmarks).
        Titik koordinat (x, y) ditransformasi secara matematis menggunakan matriks affine
        yang sama persis dengan yang diterapkan ke gambar, sehingga menghindari deteksi ulang!
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid numpy ndarray.")
        if not isinstance(landmarks, FaceLandmarkerResult):
            raise ValueError("Input landmarks must be a valid FaceLandmarkerResult instance.")
        if not landmarks.face_landmarks:
            raise ValueError("No face detected in the image.")

        lm = landmarks.face_landmarks[0]
        img_h, img_w = image.shape[:2]
        out_w, out_h = self.output_size if self.output_size else (img_w, img_h)

        left_eye = self._eye_center(lm, [133, 33, 159, 145], img_w, img_h)
        right_eye = self._eye_center(lm, [362, 263, 386, 374], img_w, img_h)

        delta = right_eye - left_eye
        angle_rad = np.arctan2(delta[1], delta[0])
        eye_dist = float(np.linalg.norm(delta)) + 1e-8
        target_eye_dist = self.eye_width_ratio * out_w
        scale = target_eye_dist / eye_dist

        ca = np.cos(-angle_rad) * scale
        sa = np.sin(-angle_rad) * scale

        midpoint_src = (left_eye + right_eye) / 2.0
        target_mid = np.array([out_w * 0.5, out_h * self.eye_center_ratio], dtype=np.float64)

        tx = target_mid[0] - (ca * midpoint_src[0] - sa * midpoint_src[1])
        ty = target_mid[1] - (sa * midpoint_src[0] + ca * midpoint_src[1])

        M = np.array([[ca, -sa, tx], [sa, ca, ty]], dtype=np.float64)

        aligned = cv2.warpAffine(
            image, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # ----------------------------------------------------------------
        # 6. Terapkan transformasi M ke koordinat landmarks
        # ----------------------------------------------------------------
        class _TransformedLandmark:
            __slots__ = ('x', 'y', 'z')
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                
        new_landmarks = []
        for point in lm:
            # Konversi normalized ke pixel awal
            px = point.x * img_w
            py = point.y * img_h
            
            # Kalikan dengan matriks Affine M
            new_px = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            new_py = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            
            # Konversi kembali ke normalized terhadap ukuran akhir
            new_landmarks.append(_TransformedLandmark(new_px / out_w, new_py / out_h, point.z))
            
        class _TransformedResult:
            __slots__ = ('face_landmarks',)
            def __init__(self, lms):
                self.face_landmarks = [lms]
                
        return aligned, _TransformedResult(new_landmarks)
