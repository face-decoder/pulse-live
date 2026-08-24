import cv2
import numpy as np


def decode_jpeg(payload: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


class Image:
    def grayscale(self, image: np.ndarray) -> np.ndarray:

        if not isinstance(image, np.ndarray) or image is None:
            raise ValueError("Input image is invalid.")

        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        elif len(image.shape) == 2:
            return image

        else:
            raise ValueError("Input image must be either a grayscale or BGR image.")

    def grayscale_gpu(
        self,
        gpu_image: "cv2.cuda_GpuMat",
        dst: "cv2.cuda_GpuMat" = None,
        stream: "cv2.cuda.Stream" = None,
    ) -> "cv2.cuda_GpuMat":
        import cv2

        if dst is None:
            dst = cv2.cuda_GpuMat()

        if stream is not None:
            cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY, dst=dst, stream=stream)
        else:
            cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY, dst=dst)

        return dst
