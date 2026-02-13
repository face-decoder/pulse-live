import cv2
import numpy as np
from src.image.modules import Image


class TVL1:

    # Image instance untuk pemrosesan gambar
    image: Image = None

    def __init__(self):
        
        # Validasi apakah CUDA tersedia pada sistem
        # Jika tersedia, gunakan versi GPU untuk pemrosesan optical flow TVL1
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            
            # Jika CUDA mendukung TVL1 maka inisialisasi model dengan itu
            if hasattr(cv2.cuda, "OpticalFlowDual_TVL1_create"):
                self.tvl1           = cv2.cuda.OpticalFlowDual_TVL1_create()
                self.gpumat_prev    = cv2.cuda_GpuMat()
                self.gpumat_next    = cv2.cuda_GpuMat()
                self.gpumat_flow    = cv2.cuda_GpuMat()

        else:

            # Jika tidak, gunakan versi CPU untuk pemrosesan optical flow TVL1
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

        # Menyesuaikan parameter TVL1 untuk mendapatkan
        # sensitivitas dan kinerja yang optimal
        self.tvl1.setLambda(0.15)
        self.tvl1.setTheta(0.3)
        self.tvl1.setTau(0.25)

        # Menginisialisasi atribut untuk menyimpan hasil optical flow
        self.flow = None

        # Menginisialisasi instance Image untuk pemrosesan gambar
        self.image = Image()


    def compute(self, prev: np.ndarray, next: np.ndarray):
        """
        Menghitung optical flow antara dua frame menggunakan algoritma TV-L1.

        Args:
            prev (numpy.ndarray): Frame sebelum.
            next (numpy.ndarray): Frame saat ini.

        Returns:
            numpy.ndarray: Optical flow yang dihitung.

        Raises:
            ValueError: Jika dimensi frame input tidak sesuai.
        """

        if prev.shape != next.shape:
            raise ValueError("Input frames must have the same dimensions.")

        # Memastikan kedua frame dalam format grayscale
        # Format ini diperlukan untuk perhitungan optical flow
        grayscalled_prev = self.image.grayscale(prev)
        grayscalled_next = self.image.grayscale(next)

        # Menghitung optical flow menggunakan TV-L1
        # Memanfaatkan GPU jika tersedia untuk mempercepat perhitungan
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, "OpticalFlowDual_TVL1_create"):

            # Mengunggah frame grayscale ke GPU
            self.gpumat_prev.upload(grayscalled_prev)
            self.gpumat_next.upload(grayscalled_next)

            # Menghitung optical flow di GPU
            self.gpumat_flow = self.tvl1.calc(self.gpumat_prev, self.gpumat_next, self.gpumat_flow)
            flow = self.gpumat_flow.download()

            # Membersihkan memori GPU
            self.gpumat_prev, self.gpumat_next, self.gpumat_next, self.gpumat_prev

        else:

            # Menghitung optical flow di CPU
            flow = self.tvl1.calc(grayscalled_prev, grayscalled_next, None)

        # Menyimpan hasil optical flow dalam atribut kelas
        self.flow = flow

        return flow