# Dokumentasi Logs API

Endpoint streaming log *real-time* dan ringkasan prediksi untuk Pulse Live.

---

## Gambaran Umum

Logs API mem-parsing file `real-time.log` dan mengekspos hasil inferensi terstruktur melalui HTTP. Seluruh data prediksi diekstrak langsung dari log — tidak memerlukan database terpisah.

Tersedia tiga endpoint:

| Endpoint | Tipe | Kegunaan |
|---|---|---|
| `GET /logs` | REST | Tail log mentah untuk debugging |
| `GET /logs/summary` | REST | N prediksi terakhir dalam format JSON |
| `GET /logs/stream` | SSE | Streaming prediksi secara *real-time* |

---

## Endpoint

### `GET /logs`

Mengembalikan N baris log mentah terakhir sebagai `text/plain`. Berguna untuk debugging.

**Parameter kueri**

| Parameter | Tipe | Default | Rentang | Deskripsi |
|---|---|---|---|---|
| `lines` | integer | `200` | 1 – 50 000 | Jumlah baris tail yang dikembalikan |

**Contoh**
```bash
# 200 baris terakhir (default)
curl http://localhost:8000/logs

# 500 baris terakhir
curl "http://localhost:8000/logs?lines=500"
```

**Respons:** `200 OK` — `text/plain`

---

### `GET /logs/summary`

Mem-parsing seluruh file log dan mengembalikan N entri **prediksi** terakhir sebagai array JSON. Melewati entri `bbox`, `heartbeat`, dan semua entri log non-prediksi.

**Parameter kueri**

| Parameter | Tipe | Default | Rentang | Deskripsi |
|---|---|---|---|---|
| `last` | integer | `20` | 1 – 500 | Jumlah prediksi terakhir yang dikembalikan |

**Contoh**
```bash
curl "http://localhost:8000/logs/summary?last=10"
```

**Respons:** `200 OK` — `application/json`

```json
[
  {
    "label": "anxiety_tinggi",
    "confidence": 0.8481,
    "detected_phases": [
      { "onset": 3, "apex": 7, "offset": 11 }
    ],
    "magnitudes": [0.12, 0.19, 0.34, 0.41, "..."],
    "smoothed_magnitudes": [0.11, 0.20, 0.33, 0.40, "..."],
    "latency_ms": 383.06
  },
  {
    "label": "anxiety_rendah",
    "confidence": 0.8005,
    "detected_phases": [],
    "magnitudes": ["..."],
    "smoothed_magnitudes": ["..."],
    "latency_ms": 421.14
  }
]
```

---

### `GET /logs/stream`

Endpoint Server-Sent Events. Saat terhubung, langsung mengirim N prediksi historis terakhir, lalu men-streaming prediksi baru secara *real-time* seiring ditulisnya ke log.

Setiap event SSE membawa satu objek JSON prediksi.

**Parameter kueri**

| Parameter | Tipe | Default | Rentang | Deskripsi |
|---|---|---|---|---|
| `history` | integer | `5` | 0 – 100 | Prediksi historis yang dikirim langsung saat terhubung |

**Contoh — JavaScript**
```js
const es = new EventSource("http://localhost:8000/logs/stream?history=5");

es.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction.label, prediction.confidence, prediction.latency_ms);
};

es.onerror = () => es.close();
```

**Contoh — curl**
```bash
curl -N "http://localhost:8000/logs/stream?history=5"
```

**Respons:** `200 OK` — `text/event-stream`

Format setiap event:
```
data: {"label":"anxiety_tinggi","confidence":0.8481,...}

```

---

## Skema Objek Prediksi

Setiap objek prediksi yang dikembalikan oleh `/logs/summary` dan `/logs/stream` memiliki field berikut:

| Field | Tipe | Deskripsi |
|---|---|---|
| `label` | `string` | Hasil klasifikasi kecemasan: `anxiety_tinggi` atau `anxiety_rendah` |
| `confidence` | `float` | Skor kepercayaan model (0.0 – 1.0) |
| `detected_phases` | `array` | Daftar fase apex yang terdeteksi. Setiap fase: `onset`, `apex`, `offset` (indeks frame) |
| `magnitudes` | `array[float]` | Magnitudo *optical flow* mentah per frame dalam window |
| `smoothed_magnitudes` | `array[float]` | Magnitudo setelah penghalusan Savitzky-Golay |
| `latency_ms` | `float` | Total latensi pipeline sejak frame pemicu tiba (ms) |

### Skema item `detected_phases`

```json
{
  "onset":  3,   // indeks frame awal fase ekspresi
  "apex":   7,   // indeks frame puncak (apex)
  "offset": 11   // indeks frame akhir fase
}
```

---

## File Log

File log `real-time.log` ditulis oleh semua modul aplikasi menggunakan modul `logging` bawaan Python.

### Format baris log

```
2026-06-20 19:24:41,652 - src.api.webrtc - INFO - <message>
```

| Field | Contoh | Deskripsi |
|---|---|---|
| Timestamp | `2026-06-20 19:24:41,652` | Waktu lokal server |
| Module | `src.api.webrtc` | Modul sumber |
| Level | `INFO` / `WARNING` / `ERROR` | Level log |
| Message | `Sending response to websocket:` | Isi log |

### Tipe pesan penting

| Awalan pesan | Arti |
|---|---|
| `Frame processing:` | Latensi per frame (WebRTC + landmark) |
| `Optical flow (TV-L1) calculation completed.` | Latensi optical flow |
| `Triggering background model inference` | Window penuh, inferensi dimulai |
| `Inference completed:` | Ringkasan latensi + label/kepercayaan prediksi |
| `Sending response to websocket:` | Payload JSON lengkap yang dikirim ke klien (di-parsing oleh API ini) |

---

## Catatan

- File log dibuka dalam **mode append** (`mode="a"`) setiap kali server dijalankan. Entri lama tetap dipertahankan. Hapus (`truncate`) `real-time.log` sebelum menjalankan server jika Anda menginginkan log bersih.
- Endpoint `/logs/stream` (SSE) melakukan polling baris baru setiap **0,3 detik**.
- Endpoint ini **tidak memiliki autentikasi**. Tambahkan middleware jika diekspos ke publik.
