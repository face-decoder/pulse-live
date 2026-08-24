import cv2
import numpy as np

from src.image.modules import Image


class TVL1:
    image: Image = None

    def __init__(self, fast_mode: bool = True):

        self.image = Image()

        self.flow = None

        self.use_cuda = False

        has_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        has_cuda_tvl1 = has_cuda and hasattr(cv2.cuda, "OpticalFlowDual_TVL1_create")

        if has_cuda_tvl1:
            self.use_cuda = True
            self.tvl1 = cv2.cuda.OpticalFlowDual_TVL1_create()

            self.gpumat_prev = cv2.cuda_GpuMat()
            self.gpumat_next = cv2.cuda_GpuMat()
            self.gpumat_flow = cv2.cuda_GpuMat()
        else:
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

        self.tvl1.setLambda(0.15)
        self.tvl1.setTheta(0.3)
        self.tvl1.setTau(0.25)

        if fast_mode:
            if hasattr(self.tvl1, "setScalesNumber"):
                self.tvl1.setScalesNumber(3)
            elif hasattr(self.tvl1, "setNumScales"):
                self.tvl1.setNumScales(3)

            if hasattr(self.tvl1, "setWarpingsNumber"):
                self.tvl1.setWarpingsNumber(2)
            elif hasattr(self.tvl1, "setNumWarps"):
                self.tvl1.setNumWarps(2)

            if hasattr(self.tvl1, "setInnerIterations"):
                self.tvl1.setInnerIterations(20)
            if hasattr(self.tvl1, "setOuterIterations"):
                self.tvl1.setOuterIterations(5)
            if hasattr(self.tvl1, "setNumIterations") and not hasattr(
                self.tvl1, "setInnerIterations"
            ):
                self.tvl1.setNumIterations(100)

            if hasattr(self.tvl1, "setMedianFiltering"):
                self.tvl1.setMedianFiltering(1)
        else:
            if hasattr(self.tvl1, "setScalesNumber"):
                self.tvl1.setScalesNumber(5)
            elif hasattr(self.tvl1, "setNumScales"):
                self.tvl1.setNumScales(5)

            if hasattr(self.tvl1, "setWarpingsNumber"):
                self.tvl1.setWarpingsNumber(5)
            elif hasattr(self.tvl1, "setNumWarps"):
                self.tvl1.setNumWarps(5)

            if hasattr(self.tvl1, "setInnerIterations"):
                self.tvl1.setInnerIterations(30)
            if hasattr(self.tvl1, "setOuterIterations"):
                self.tvl1.setOuterIterations(10)
            if hasattr(self.tvl1, "setNumIterations") and not hasattr(
                self.tvl1, "setInnerIterations"
            ):
                self.tvl1.setNumIterations(300)

    def __prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = frame if frame.ndim == 2 else self.image.grayscale(frame)

        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8, copy=False)

        return np.ascontiguousarray(gray)

    def compute(self, prev: np.ndarray, next: np.ndarray, download: bool = True):
        if prev is None or next is None:
            raise ValueError("Input frame is None.")
        if prev.shape != next.shape:
            raise ValueError("Input frames must have the same dimensions.")

        gray_prev = self.__prepare_gray(prev)
        gray_next = self.__prepare_gray(next)

        if self.use_cuda:
            self.gpumat_prev.upload(gray_prev)
            self.gpumat_next.upload(gray_next)

            self.gpumat_flow = self.tvl1.calc(
                self.gpumat_prev, self.gpumat_next, self.gpumat_flow
            )
            flow = self.gpumat_flow.download() if download else self.gpumat_flow
        else:
            flow = self.tvl1.calc(gray_prev, gray_next, None)

        self.flow = flow
        return flow

    def compute_batch(self, frame_pairs: list, download: bool = True) -> list:
        results = []
        for prev, next_frame in frame_pairs:
            flow = self.compute(prev, next_frame, download=download)
            results.append(flow)
        return results

    def close(self):
        if self.use_cuda:
            self.gpumat_prev.release()
            self.gpumat_next.release()
            self.gpumat_flow.release()
