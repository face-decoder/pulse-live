# Panduan Build Manual MediaPipe Python dengan Dukungan GPU (Linux)

Secara bawaan (*default*), pustaka `mediapipe` yang di-*install* melalui `pip` di Linux dikompilasi tanpa dukungan GPU (`GPU processing is disabled in build flags`). 

Jika Anda ingin memaksimalkan *Face Landmark* agar menggunakan CUDA (GPU), Anda harus melakukan kompilasi mandiri dari C++ *source code* menggunakan **Bazel**. 

Panduan ini akan memandu Anda melakukan *build* untuk menghasilkan *installer* Python (`.whl`) khusus dengan CUDA teraktivasi.

---

## 1. Persyaratan Sistem (*Prerequisites*)

Pastikan komputer/server Linux Anda sudah memiliki dependensi berikut:
1. **NVIDIA GPU** dengan *driver* terbaru.
2. **CUDA Toolkit** (disarankan 11.8 atau 12.x) beserta **cuDNN**.
3. **GCC / G++** versi 9 atau lebih baru.
4. **Python 3.8+** beserta lingkungan virtual (*virtual environment*).
5. **Bazelisk** (Pengelola versi Bazel otomatis dari Google).

Install *library* grafis yang dibutuhkan untuk kompilasi (OpenGL & EGL):
```bash
sudo apt-get update
sudo apt-get install -y mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
```

---

## 2. Persiapan Repositori & Environment

Lakukan *clone* pada repositori resmi MediaPipe dan arahkan *path* CUDA:

```bash
# 1. Clone repositori
git clone https://github.com/google/mediapipe.git
cd mediapipe

# 2. Atur Path CUDA (Ganti /usr/local/cuda-12.2 sesuai versi CUDA Anda)
export TF_CUDA_PATHS=/usr/local/cuda-12.2

# 3. Install paket python yang dibutuhkan untuk build
pip install setuptools wheel
pip install -r requirements.txt
```

---

## 3. Proses Kompilasi (*Build*)

Jalankan perintah berikut untuk menginstruksikan Bazel membuat *wheel* Python (`.whl`). 
> **Peringatan:** Proses ini memakan waktu **1 hingga 3 jam** tergantung jumlah *core* CPU komputer Anda dan akan mengonsumsi banyak RAM.

```bash
python3 setup.py bdist_wheel \
    --link-opencv \
    --bazel-flags="--config=cuda --copt=-DMESA_EGL_NO_X11_HEADERS"
```
*Catatan: Flag `--config=cuda` adalah parameter utama untuk mengaktifkan TensorFlow CUDA delegate yang dibutuhkan oleh MediaPipe GPU pada Linux.*

---

## 4. Mengambil Hasil Build dan Meng-installnya

Jika proses kompilasi berhasil tanpa error, Bazel akan menghasilkan satu file berformat `.whl` (Wheel) yang tersimpan otomatis di dalam folder `dist/`.

Contoh nama file: `mediapipe-0.10.x-cp310-cp310-linux_x86_64.whl`

### Cara Install ke Project Anda:
```bash
# Masuk ke environment project Anda
cd /path/ke/project/anda
source .venv/bin/activate

# Hapus mediapipe versi CPU (jika ada)
pip uninstall mediapipe -y

# Install file hasil build yang baru saja dibuat
pip install /path/ke/mediapipe/dist/mediapipe-*.whl
```

### Cara Menyimpan/Memindah Hasil Build:
File `.whl` yang ada di dalam `dist/` ini bersifat portabel untuk versi Linux dan Python yang sama. Anda cukup **menyalin (copy)** file tersebut ke *flashdisk* atau komputer lain, dan bisa langsung menginstalnya dengan perintah `pip install` tanpa perlu mengulang proses *build* berjam-jam.

---

## 5. Implementasi pada Kode (Python)

Setelah berhasil di-*install*, Anda wajib menambahkan konfigurasi `delegate` pada `BaseOptions` di kode Python agar *engine* MediaPipe benar-benar menggunakan GPU.

```python
import mediapipe as mp
from mediapipe.tasks import python

base_options = python.BaseOptions(
    model_asset_path='models/face_landmarker.task',
    delegate=python.BaseOptions.Delegate.GPU  # Aktifkan GPU Delegate
)

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)

landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
print("GPU Face Landmarker berhasil diinisialisasi!")
```
