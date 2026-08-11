# Video Upload WebSocket API

Protokol integrasi WebSocket untuk Frontend (FE) dalam mengirimkan aliran video.

## 1. Koneksi
`ws://<host>/ws/video/{session_id}`
- `session_id`: String unik identifier sesi (e.g. UUID).

## 2. Client → Server (Sisi FE)
Kirim data dengan urutan berikut:

1. **Inisialisasi (Teks / JSON)**
   ```json
   {
     "type": "start",
     "filename": "video.webm",
     "size": 1024000
   }
   ```
2. **Video Chunks (Biner / ArrayBuffer)**
   Kirim *chunk* video secara kontinu. 
   > **[INFO]** Gunakan format `webm` (MediaRecorder) atau *Fragmented MP4* (fMP4) agar server bisa menganalisis gambar seketika (*real-time*). Format `mp4` reguler akan *stuck* menunggu file selesai.
3. **Selesai (Teks / JSON)**
   ```json
   {
     "type": "end"
   }
   ```

## 3. Server → Client (Respons JSON)
Selama streaming, FE akan menerima event berikut secara *real-time*:

**A. Status / Info**
```json
{
  "type": "status",
  "status": "receiving",  // atau "completed"
  "message": "..."
}
```

**B. Hasil Deteksi / Prediksi**
```json
{
  "type": "prediction",
  "label": "normal",
  "confidence": 0.95,
  "face_bboxes": [{"x": 10.0, "y": 20.0, "width": 100.0, "height": 100.0}]
}
```

**C. Error**
```json
{
  "type": "error",
  "message": "..."
}
```

**D. Heartbeat** (Abaikan, untuk menjaga koneksi tetap hidup)
```json
{
  "type": "heartbeat"
}
```

**E. Summary (Rolling Update)**
Setiap kali ada hasil inferensi baru, server juga akan langsung mengirimkan ringkasan kalkulasi **keseluruhan durasi video sejauh ini**. FE dapat menggunakannya untuk *update* progres dan me-*render* grafik magnitudo secara *real-time*:
```json
{
  "type": "summary",
  "data": {
    "total_windows": 12,
    "anxiety_detected": 3,
    "avg_confidence": 0.8845,
    "magnitudes": [0.1, 0.5, 0.4, 1.2, 0.2],
    "smoothed_magnitudes": [0.12, 0.45, 0.42, 1.1, 0.25],
    "detected_phases": [
      {
        "onset": 1,
        "apex": 3,
        "offset": 4
      }
    ]
  }
}
```
