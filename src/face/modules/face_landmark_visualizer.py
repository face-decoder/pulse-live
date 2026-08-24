import cv2
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections


class FaceLandmarkVisualizer:
    def __init__(self, landmark_result):  # type: ignore

        if not hasattr(landmark_result, "face_landmarks"):
            raise ValueError("landmark_result must have 'face_landmarks' attribute.")

        self.landmark_result = landmark_result

    def draw(self, image: np.ndarray, connections="tesselation") -> np.ndarray:

        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray.")

        if image.size == 0:
            raise ValueError("Input image is empty.")

        if connections not in ["tesselation", "contours"]:
            raise ValueError("Connections must be either 'tesselation' or 'contours'.")

        if connections == "contours":
            connection_set = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
        else:
            connection_set = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

        h, w, _ = image.shape

        for normalized_landmark in self.landmark_result.face_landmarks:
            for connection in connection_set:
                start_idx = connection.start
                end_idx = connection.end

                if start_idx < len(normalized_landmark) and end_idx < len(
                    normalized_landmark
                ):
                    start_point = normalized_landmark[start_idx]
                    end_point = normalized_landmark[end_idx]

                    p1 = (int(start_point.x * w), int(start_point.y * h))
                    p2 = (int(end_point.x * w), int(end_point.y * h))

                    cv2.line(image, p1, p2, (255, 0, 0), 1)

            for point in normalized_landmark:
                x_px = int(point.x * w)
                y_px = int(point.y * h)
                cv2.circle(image, (x_px, y_px), 1, (0, 255, 0), -1)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
