import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult


class FaceAlignerPoints:

    # Titik mata kiri atas
    LEFT_EYE_TOP = 159

    # Titik mata kanan atas
    RIGHT_EYE_TOP = 386

    # Titik ujung hidung
    NOSE_TIP = 1

    # Titik mulut kiri
    MOUTH_LEFT = 61

    # Titik mulut kanan
    MOUTH_RIGHT = 291


class FaceAligner:
    
    def align(self, image: np.ndarray = None, landmarks: FaceLandmarkerResult = None) -> np.ndarray:
        """
        Menyelaraskan wajah dalam citra berdasarkan titik landmark yang sudah didapat.
        
        Args:
            image (np.ndarray): Citra input yang berisi wajah yang akan diselaraskan.
            landmarks (FaceLandmarkerResult): Hasil deteksi landmark wajah.
            
        Returns:
            np.ndarray: Citra wajah yang sudah diselaraskan.

        Raises:
            ValueError: Jika tidak ada wajah yang terdeteksi dalam citra.
        """
        
        # Melakukan validasi input citra untuk memastikan adalah instance dari numpy ndarray
        # Jika tidak, maka akan mengembalikan exception ValueError
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a valid numpy ndarray.")
        
        # Melakukan validasi input landmark untuk memastikan adalah instance dari FaceLandmarkerResult
        # Jika tidak, maka akan mengembalikan exception ValueError
        if not isinstance(landmarks, FaceLandmarkerResult):
            raise ValueError("Input landmarks must be a valid FaceLandmarkerResult instance.")
        
        # Memastikan bahwa ada wajah yang terdeteksi dalam citra
        # Jika tidak, maka akan mengembalikan exception ValueError
        if not landmarks.face_landmarks:
            raise ValueError("No face detected in the image.")
        
        # Mengambil landmark dari wajah pertama yang terdeteksi
        landmark = landmarks.face_landmarks[0]
        
        h, w = image.shape[:2]
        
        # Mendefinisikan titik referensi untuk penyelarasan wajah
        reference_points = np.array([
            [landmark[FaceAlignerPoints.LEFT_EYE_TOP].x * w, landmark[FaceAlignerPoints.LEFT_EYE_TOP].y * h],
            [landmark[FaceAlignerPoints.RIGHT_EYE_TOP].x * w, landmark[FaceAlignerPoints.RIGHT_EYE_TOP].y * h],
            [landmark[FaceAlignerPoints.NOSE_TIP].x * w, landmark[FaceAlignerPoints.NOSE_TIP].y * h],
        ], dtype=np.float32)
        
        # Titik tujuan ini adalah posisi standar di mana titik referensi
        # harus ditempatkan setelah penyelarasan
        # Menggunakan proporsi yang lebih natural untuk wajah
        destination_points = np.array([
            [0.35 * w, 0.35 * h],  # Left eye   -> lebih ke tengah
            [0.65 * w, 0.35 * h],  # Right eye  -> lebih ke tengah
            [0.5 * w, 0.55 * h],   # Nose tip   -> sedikit di bawah mata
        ], dtype=np.float32)
        
        # Menghitung matriks transformasi afine untuk penyelarasan wajah
        # Hanya menggunakan 3 titik pertama (kedua mata dan hidung)
        # karena cv2.getAffineTransform() hanya membutuhkan 3 titik
        transformation_matrix = cv2.getAffineTransform(
            reference_points, 
            destination_points
        )
        
        # Menerapkan transformasi dengan parameter interpolasi linear
        # dan border mode replicate untuk mengisi area kosong
        aligned_face = cv2.warpAffine(image, 
                                      transformation_matrix, 
                                      (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        
        return aligned_face