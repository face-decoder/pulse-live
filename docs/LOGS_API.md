# Logs API

Tiga endpoint untuk mengakses `real-time.log`.

## Endpoints

### `GET /logs?lines=N`

Mengembalikan **N baris terakhir** raw log sebagai plain-text (untuk debugging).

| Parameter | Default | Range |
|---|---|---|
| `lines` | `200` | 1 – 50 000 |

```bash
curl http://localhost:8000/logs
```

---

### `GET /logs/summary?last=N`

Parse log dan kembalikan **N prediksi terakhir** sebagai JSON array. Hanya memuat field yang relevan.

| Parameter | Default | Range |
|---|---|---|
| `last` | `20` | 1 – 500 |

**Response contoh:**
```json
[
  {
    "label": "anxiety_tinggi",
    "confidence": 0.8481,
    "detected_phases": [
      { "onset": 3, "apex": 7, "offset": 11 }
    ],
    "magnitudes": [0.12, 0.34, ...],
    "smoothed_magnitudes": [0.11, 0.33, ...],
    "latency_ms": 383.06
  }
]
```

```bash
curl "http://localhost:8000/logs/summary?last=10"
```

---

### `GET /logs/stream?history=N`

Server-Sent Events: kirim **N prediksi historis langsung saat connect**, lalu streaming prediksi baru secara real-time.

Setiap event adalah JSON object dengan field yang sama seperti `/logs/summary`.

| Parameter | Default | Range |
|---|---|---|
| `history` | `5` | 0 – 100 |

**Contoh — JavaScript**
```js
const es = new EventSource("http://localhost:8000/logs/stream?history=5");
es.onmessage = (e) => {
  const pred = JSON.parse(e.data);
  console.log(pred.label, pred.confidence, pred.latency_ms);
};
```

**Contoh — curl**
```bash
curl -N "http://localhost:8000/logs/stream?history=5"
```

---

## Field Prediksi

| Field | Tipe | Keterangan |
|---|---|---|
| `label` | `string` | Hasil klasifikasi (`anxiety_tinggi` / `anxiety_rendah`) |
| `confidence` | `float` | Skor kepercayaan model (0–1) |
| `detected_phases` | `array` | Fase apex yang terdeteksi: `onset`, `apex`, `offset` (index frame) |
| `magnitudes` | `array[float]` | Magnitudo optical flow per frame |
| `smoothed_magnitudes` | `array[float]` | Magnitudo setelah Savitzky-Golay smoothing |
| `latency_ms` | `float` | Total latensi pipeline dari frame trigger (ms) |

---

## Catatan

- `/logs/summary` dan `/logs/stream` hanya mengambil entry `"type": "prediction"` dari log — melewati `bbox`, `heartbeat`, dan log non-JSON.
- File log di-append tiap run. Untuk mulai bersih, truncate `real-time.log` sebelum server dijalankan.
