import cv2


class Video:
    def __init__(self, video_path: str) -> None:

        self.video_path = video_path

        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

    def read_all_frames(self) -> list:
        frames = []

        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            frames.append(frame)

        self.capture.release()
        return frames

    def get_frame_pairs(self) -> list:
        frames = self.read_all_frames()
        pairs = []
        for i in range(len(frames) - 1):
            pairs.append((frames[i], frames[i + 1], i))
        return pairs

    def map(self, func: callable) -> list:
        results = []
        frame_idx = 0

        ret, prev_frame = self.capture.read()
        if not ret:
            self.capture.release()
            return results

        while True:
            ret, curr_frame = self.capture.read()
            if not ret:
                break

            result = func(prev_frame, curr_frame, frame_idx)
            results.append(result)

            prev_frame = curr_frame
            frame_idx += 1

        self.capture.release()
        return results
