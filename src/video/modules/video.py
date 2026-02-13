import cv2


class Video:

    def __init__(self, video_path: str) -> None:
        """
        Inisialisasi objek Video dengan path video yang diberikan.

        Args:
            video_path (str): Path ke file video.

        Raises:
            ValueError: Jika file video tidak dapat dibuka.
        """
        
        # Menyimpan path video dan membuka file video menggunakan OpenCV
        self.video_path = video_path

        # Membuka file video menggunakan OpenCV
        self.capture = cv2.VideoCapture(video_path)

        # Jika file video tidak dapat dibuka
        # Kembalikan dengan raise ValueError
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")


    def map(self, func: callable) -> list:
        """
        Menerapkan fungsi pada setiap pasangan frame berturut-turut dalam video.

        Args:
            func (callable): Fungsi yang akan diterapkan pada setiap pasangan frame.
                             Fungsi ini harus menerima dua argumen (frame sebelumnya dan frame saat ini)
                             dan mengembalikan hasil yang diinginkan.

        Returns:
            list: Daftar hasil yang dikembalikan oleh fungsi untuk setiap pasangan frame.
        """
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
