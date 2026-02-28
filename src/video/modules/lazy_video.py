import cv2


class LazyVideo:

    def __init__(self, video_path: str):
        """
        Inisialisasi objek LazyVideo dengan path video yang diberikan.

        Args:
            video_path (str): Path ke file video.

        Raises:
            ValueError: Jika file video tidak dapat dibuka.
        """
        
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        # Jika file video tidak dapat dibuka, mengembalikan dengan raise ValueError
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Mengambil metadata video untuk optimasi akses frame
        self.count      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame dalam video
        self.width      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Lebar frame video
        self.height     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Tinggi frame video
        self.fps        = self.cap.get(cv2.CAP_PROP_FPS)               # Frame per detik video, berguna untuk timing dan sinkronisasi


    def __len__(self) -> int:
        """
        Mengembalikan jumlah frame dalam video. Ini memungkinkan penggunaan len() pada objek LazyVideo untuk mendapatkan total frame.

        Returns:
            int: Jumlah frame dalam video.
        """
        return self.count


    def __getitem__(self, idx: int | slice):
        """
        Mengakses frame video secara acak berdasarkan indeks. Ini memungkinkan penggunaan slicing dan indexing pada objek LazyVideo untuk mendapatkan frame tertentu.

        Args:
            idx (int or slice): Indeks frame yang ingin diakses. Bisa berupa integer untuk frame tunggal atau slice untuk rentang frame.

        Returns:
            numpy.ndarray or list: Frame video sebagai array numpy jika idx adalah integer, atau daftar frame jika idx adalah slice.
        """

        # Jika index adalah slice, kembalikan list frame yang sesuai dengan slice tersebut
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        
        # Jika index adalah integer, kembalikan frame tunggal
        if idx < 0: idx += len(self)
            
        # Jika index di luar batas, raise IndexError
        # Ini menunjukkan bahwa akses frame yang diminta tidak valid, sehingga mencegah akses ke frame yang tidak ada dalam video.
        if idx >= len(self) or idx < 0:
            raise IndexError("Video frame index out of range")

        # Set posisi frame pada video ke indeks yang diminta dan baca frame tersebut
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
        """
        Menutup video capture untuk membebaskan sumber daya. 
        Ini penting untuk mencegah kebocoran memori dan memastikan bahwa file video tidak terkunci setelah selesai digunakan.
        """
        self.cap.release()
    
    def __del__(self):
        """
        Memastikan bahwa video capture ditutup saat objek LazyVideo dihapus dari memori. 
        Ini adalah langkah pencegahan tambahan untuk memastikan bahwa sumber daya dibersihkan dengan benar, 
        bahkan jika pengguna lupa memanggil metode close() secara eksplisit.
        """
        self.close()
