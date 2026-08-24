import cv2


class LazyVideo:
    def __init__(self, video_path: str):

        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        raw_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if raw_count > 0:
            self.count = raw_count
        else:
            n = 0
            while True:
                ret, _ = self.cap.read()
                if not ret:
                    break
                n += 1
            self.count = n
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int | slice):

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        if idx < 0:
            idx += len(self)

        if idx >= len(self) or idx < 0:
            raise IndexError("Video frame index out of range")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = self.cap.read()

        if not ret:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()

            if not ret:
                raise ValueError(f"Could not read frame {idx} from {self.video_path}")

        return frame

    def close(self):
        self.cap.release()

    def __del__(self):
        self.close()
