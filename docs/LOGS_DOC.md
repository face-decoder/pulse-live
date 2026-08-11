# Logs API Documentation

Real-time log streaming and prediction summary endpoints for Pulse Live.

---

## Overview

The Logs API parses `real-time.log` and exposes structured inference results over HTTP. All prediction data is extracted directly from the log — no separate database required.

Three endpoints are available:

| Endpoint | Type | Use case |
|---|---|---|
| `GET /logs` | REST | Raw log tail for debugging |
| `GET /logs/summary` | REST | Last N parsed predictions as JSON |
| `GET /logs/stream` | SSE | Live prediction stream |

---

## Endpoints

### `GET /logs`

Returns the last N raw log lines as `text/plain`. Useful for debugging.

**Query parameters**

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `lines` | integer | `200` | 1 – 50 000 | Number of tail lines to return |

**Example**
```bash
# Last 200 lines (default)
curl http://localhost:8000/logs

# Last 500 lines
curl "http://localhost:8000/logs?lines=500"
```

**Response:** `200 OK` — `text/plain`

---

### `GET /logs/summary`

Parses the full log file and returns the last N **prediction** entries as a JSON array. Skips `bbox`, `heartbeat`, and all non-prediction log entries.

**Query parameters**

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `last` | integer | `20` | 1 – 500 | Number of recent predictions to return |

**Example**
```bash
curl "http://localhost:8000/logs/summary?last=10"
```

**Response:** `200 OK` — `application/json`

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

Server-Sent Events endpoint. On connect, immediately sends the last N historical predictions, then streams new predictions in real-time as they are written to the log.

Each SSE event carries a single prediction JSON object.

**Query parameters**

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `history` | integer | `5` | 0 – 100 | Historical predictions to send immediately on connect |

**Example — JavaScript**
```js
const es = new EventSource("http://localhost:8000/logs/stream?history=5");

es.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction.label, prediction.confidence, prediction.latency_ms);
};

es.onerror = () => es.close();
```

**Example — curl**
```bash
curl -N "http://localhost:8000/logs/stream?history=5"
```

**Response:** `200 OK` — `text/event-stream`

Each event format:
```
data: {"label":"anxiety_tinggi","confidence":0.8481,...}

```

---

## Prediction Object Schema

Every prediction object returned by `/logs/summary` and `/logs/stream` contains these fields:

| Field | Type | Description |
|---|---|---|
| `label` | `string` | Anxiety classification result: `anxiety_tinggi` or `anxiety_rendah` |
| `confidence` | `float` | Model confidence score (0.0 – 1.0) |
| `detected_phases` | `array` | List of detected apex phases. Each phase: `onset`, `apex`, `offset` (frame indices) |
| `magnitudes` | `array[float]` | Raw optical flow magnitude per frame in the window |
| `smoothed_magnitudes` | `array[float]` | Magnitudes after Savitzky-Golay smoothing |
| `latency_ms` | `float` | Total pipeline latency from trigger frame arrival (ms) |

### `detected_phases` item schema

```json
{
  "onset":  3,   // frame index where the expression phase begins
  "apex":   7,   // frame index of the peak (apex)
  "offset": 11   // frame index where the phase ends
}
```

---

## Log File

The log file `real-time.log` is written by all application modules using Python's standard `logging` module.

### Log line format

```
2026-06-20 19:24:41,652 - src.api.webrtc - INFO - <message>
```

| Field | Example | Description |
|---|---|---|
| Timestamp | `2026-06-20 19:24:41,652` | Server local time |
| Module | `src.api.webrtc` | Source module |
| Level | `INFO` / `WARNING` / `ERROR` | Log level |
| Message | `Sending response to websocket:` | Log content |

### Key message types

| Message prefix | Meaning |
|---|---|
| `Frame processing:` | Per-frame latency (WebRTC + landmark) |
| `Optical flow (TV-L1) calculation completed.` | Optical flow latency |
| `Triggering background model inference` | Window full, inference started |
| `Inference completed:` | Latency summary + prediction label/confidence |
| `Sending response to websocket:` | Full JSON payload sent to client (parsed by this API) |

---

## Notes

- The log file is opened in **append mode** (`mode="a"`) on every server start. Old entries are preserved. Truncate `real-time.log` before starting the server if you want a clean log.
- The `/logs/stream` SSE endpoint polls for new lines every **0.3 seconds**.
- There is **no authentication** on these endpoints. Add middleware if exposing to the public.
