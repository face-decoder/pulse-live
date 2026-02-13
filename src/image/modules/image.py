import cv2
import numpy as np


class Image:

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Mengonversi gambar input menjadi format grayscale.

        Args:
            image (numpy.ndarray): Citra input .

        Returns:
            numpy.ndarray: Citra grayscale.

        Raises:
            None
        """

        # Memastikan input adalah image yang valid.
        if not isinstance(image, np.ndarray) or image is None:
            raise ValueError("Input image is invalid.")

        # Validasi dan konversi gambar ke format grayscale
        # Jika gambar berwarna (BGR) atau (RGB), konversi ke grayscale
        if len(image.shape) == 3 and image.shape[2] == 3: 
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Jika gambar sudah dalam format grayscale, kembalikan langsung
        elif len(image.shape) == 2: return image

        # Jika format gambar tidak dikenali, kembalikan dengan raise ValueError
        else:
            raise ValueError("Input image must be either a grayscale or BGR image.")