# Pulse Live

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.8-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA_12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-MinIO_S3-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](file:///home/inadio/skripkir/pulse-live/LICENSE)

Real-time facial micro-expression detection, phase-based apex spotting, and deep learning analysis platform using computer vision, optical flow, and WebRTC streaming.

---

## Table of Contents

- [Project Description](#project-description)
  - [Overview](#overview)
  - [Technology Rationale](#technology-rationale)
  - [Challenges & Solutions](#challenges--solutions)
  - [Roadmap](#roadmap)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation](#step-by-step-installation)
- [Usage Guide](#usage-guide)
  - [Real-Time WebRTC / WebSocket Streaming](#real-time-webrtc--websocket-streaming)
  - [Offline Video Upload](#offline-video-upload)
  - [Makefile Commands](#makefile-commands)
  - [System Architecture](#system-architecture)
- [Testing & Benchmarks](#testing--benchmarks)
  - [Running Spotting Benchmarks](#running-spotting-benchmarks)
  - [Generating Visualizations](#generating-visualizations)
  - [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Credits & Acknowledgements](#credits--acknowledgements)
- [License](#license)

---

## Project Description

### Overview

**Pulse Live** is an end-to-end computer vision and deep learning platform for real-time and offline detection, temporal apex spotting, and classification of facial micro-expressions. Micro-expressions are brief, involuntary facial movements (lasting 1/25s to 1/5s) that expose subtle psychological responses.

The automated multi-stage pipeline consists of:
1. **Landmark Alignment**: Real-time 468-point tracking via MediaPipe FaceMesh to extract facial Regions of Interest (ROIs).
2. **Phase-Based Apex Spotting**: Detection of micro-movement onset, apex (peak intensity), and offset frames using Riesz Pyramids.
3. **Dense Optical Flow**: CUDA-accelerated TV-L1 optical flow extraction powered by CuPy.
4. **Deep Learning Classification**: Sequence modeling (CNN-Transformer, Bi-LSTM, TCN, 3D CNN) with Test-Time Augmentation (TTA).
5. **Streaming & Storage**: Low-latency WebRTC (`aiortc`) and WebSocket telemetry combined with MinIO S3 object storage.

---

### Technology Rationale

- **Python 3.12 & `uv`**: Fast execution and reproducible environment management.
- **FastAPI & `aiortc` / WebSockets**: Asynchronous, low-latency video stream ingestion and real-time telemetry streaming.
- **CUDA-Accelerated MediaPipe & OpenCV**: Custom GPU wheels to offload landmark alignment and frame rendering from the CPU.
- **CuPy & TV-L1 Optical Flow**: GPU-accelerated spatial-temporal motion vector computation for high-frame-rate processing.
- **PyTorch (CUDA 12.8)**: High-performance neural network evaluation supporting modern sequence architectures.
- **MinIO S3**: Scalable object storage for raw session recordings, landmark metadata, and optical flow artifacts.

---

### Challenges & Solutions

- **Subtle Motion Detection**: Overcame high noise in micro-movements by combining Riesz Pyramids with dense TV-L1 optical flow.
- **Real-Time Pipeline Sync**: Prevented frame drops under heavy GPU model evaluation through non-blocking async buffer queues.
- **Custom GPU Build Dependencies**: Maintained custom Linux GPU wheels for MediaPipe 0.10.15 and OpenCV CUDA 4.15.

---

### Roadmap

- [ ] **Multi-Person Tracking**: Concurrent multi-face ROI isolation in dense video frames.
- [ ] **TensorRT / ONNX Acceleration**: Sub-millisecond model optimization for edge deployment.
- [ ] **Web Dashboard**: Interactive React/Next.js interface for live telemetry and analytics.
- [ ] **Multimodal Fusion**: Combined audio prosody and facial optical flow analysis.

---

## Key Features

- **Live Streaming API**: Low-latency WebRTC stream ingestion and WebSocket JSON telemetry delivery.
- **Offline Processing**: REST API endpoints for batch video feature extraction, apex spotting, and data export.
- **Phase Apex Spotting**: Riesz Pyramid phase dynamics to isolate peak micro-expression frames.
- **Targeted ROI Cropping**: Automatic extraction of key facial regions (eyebrows, eyes, nose, mouth).
- **GPU Optical Flow**: High-throughput TV-L1 motion vectors computed via custom CUDA/CuPy kernels.
- **Deep Learning Suite**: CNN-Transformer, Bi-LSTM (Attention), TCN, and 3D Spatio-Temporal networks with TTA.
- **MinIO Object Storage**: Automated persistence for videos, landmark JSONs, and optical flow maps.
- **Telemetry & Logs API**: Endpoints for retrieving execution logs, session histories, and runtime metrics.

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch (CUDA 12.8), Torchvision, SciPy, NumPy, Scikit-Learn |
| **GPU Acceleration** | CuPy (CUDA 12.x), CUDA Toolkit 12.8 |
| **Computer Vision** | OpenCV CUDA 4.15.0, MediaPipe 0.10.15 (Custom GPU Wheel), PyAV (`av`) |
| **Web & Async I/O** | FastAPI, Uvicorn, WebSockets, `aiortc` (WebRTC) |
| **Object Storage** | MinIO (S3-Compatible Storage) |
| **DevOps & Containers** | Docker, Docker Compose, Makefile |
| **Package Manager** | [`uv`](https://github.com/astral-sh/uv) |

---

## Installation & Setup

### Prerequisites

- **OS**: Linux (Ubuntu 22.04 LTS or newer recommended)
- **Python**: Version `3.12.*`
- **NVIDIA GPU**: CUDA 12.x supported GPU with Driver 12.8 installed
- **Tooling**: [`uv`](https://github.com/astral-sh/uv), Docker & Docker Compose, GNU `make`

---

### Step-by-Step Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/pulse-live.git
   cd pulse-live
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   ```
   *Note: Customize model path, MinIO credentials, and ports inside `.env`.*

3. **Install Dependencies**
   ```bash
   make sync-deps
   # Or directly:
   uv sync
   ```

4. **Launch Infrastructure**
   ```bash
   make infra
   ```

5. **Run Application Server**
   ```bash
   make dev
   ```
   Server runs at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

---

## Usage Guide

### Real-Time WebRTC / WebSocket Streaming

1. Send a WebRTC offer to `/api/webrtc/offer` to negotiate low-latency video media tracks.
2. Connect to `/api/ws/telemetry` over WebSockets to receive live frame analytics:
   - Face bounding boxes & landmark status
   - Isolated facial ROIs
   - Apex frame spotting indicators
   - Emotion classification confidence scores

See [`docs/API_CONTRACT.md`](file:///home/inadio/skripkir/pulse-live/docs/API_CONTRACT.md) and [`docs/webrtc_websocket_workflow.md`](file:///home/inadio/skripkir/pulse-live/docs/webrtc_websocket_workflow.md) for protocol details.

---

### Offline Video Upload

Process pre-recorded video files via POST request:
```bash
curl -X POST "http://localhost:8000/api/video/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/sample_video.mp4"
```
Refer to [`docs/VIDEO_UPLOAD.md`](file:///home/inadio/skripkir/pulse-live/docs/VIDEO_UPLOAD.md) for details.

---

### Makefile Commands

| Command | Description |
| :--- | :--- |
| `make run` | Run server in production mode (`uv run python main.py`) |
| `make dev` | Run development server with auto-reload |
| `make infra` | Start MinIO Docker container |
| `make infra-down` | Stop MinIO Docker container |
| `make infra-logs` | Tail MinIO Docker container logs |
| `make sync-deps` | Synchronize virtual environment dependencies |
| `make clean` | Remove temporary cache files (`__pycache__`, `.tmp`) |

---

### System Architecture

```
[ Camera / Video Input ]
          │
          ▼
 [ WebRTC Receiver ] ──► [ MediaPipe FaceMesh ] ──► [ ROI Extraction ]
                                                           │
                                                           ▼
 [ MinIO Storage ] ◄── [ TV-L1 Optical Flow ] ◄─── [ Riesz Apex Spotting ]
        ▲                       │
        │                       ▼
 [ Telemetry Log ] ◄── [ PyTorch Deep Model ] ──► [ WebSocket Output ]
```

---

## Testing & Benchmarks

Validate apex spotting and classification accuracy on standard academic datasets (CAS(ME)^2 and SAMM).

### Running Spotting Benchmarks

```bash
# CAS(ME)^2 benchmark
uv run python cas_me_2_spotting.py

# SAMM benchmark
uv run python samm_spotting.py
```

### Generating Visualizations

```bash
uv run python generate_apex_visual.py
```

### Evaluation Metrics

See documentation for evaluation mathematical logic and thresholding:
- [`docs/CONFUSION_MATRIX_MANUAL_CALC.md`](file:///home/inadio/skripkir/pulse-live/docs/CONFUSION_MATRIX_MANUAL_CALC.md): Confusion matrix verification formulas.
- [`docs/CUTOFF_RATIO_CASME_II.md`](file:///home/inadio/skripkir/pulse-live/docs/CUTOFF_RATIO_CASME_II.md): Cutoff threshold specifications for CAS(ME)^2.

---

## Project Structure

```
pulse-live/
├── .env                          # Environment variables & model configuration
├── docker-compose.yml            # Infrastructure definition (MinIO)
├── Makefile                      # Command shortcuts
├── pyproject.toml                # UV package manifest
├── main.py                       # FastAPI application entrypoint
├── docs/                         # Specifications & API documentation
│   ├── API_CONTRACT.md           # WebRTC/WebSocket contract
│   ├── BUILD_MEDIAPIPE_GPU.md    # GPU wheel compilation guide
│   ├── CONFUSION_MATRIX_MANUAL_CALC.md # Metric calculation logic
│   ├── CUTOFF_RATIO_CASME_II.md  # Temporal cutoff ratios
│   ├── LOGS_API.md               # Logging API reference
│   ├── VIDEO_UPLOAD.md           # Offline upload specification
│   ├── webrtc_websocket_workflow.md # Streaming pipeline guide
│   └── workflow.md               # System processing overview
├── packages/                     # Custom Linux GPU wheels (.whl)
│   ├── mediapipe-0.10.15-*.whl   # Custom MediaPipe wheel
│   └── opencv_cuda-4.15.0-*.whl  # Custom OpenCV CUDA wheel
├── scripts/                      # Utility scripts
├── notebooks/                    # Analysis & experiment notebooks
└── src/                          # Application source code
    ├── api/                      # FastAPI endpoints (WebRTC, WS, Video, Logs)
    ├── apex/                     # Riesz Pyramid apex spotters
    ├── dataset/                  # Dataset loaders & augmentations
    ├── datasource/               # Sequence extractors
    ├── evaluator/                # Metrics calculation
    ├── face/                     # Landmark tracking & ROI aligners
    ├── models/                   # Neural network architectures & inferencer
    ├── optical_flow/             # CUDA TV-L1 optical flow module
    ├── storage/                  # MinIO S3 storage integration
    ├── utils/                    # Signal processing helpers
    └── video/                    # Frame extraction & video decoders
```

---

## Contributing

Direct public contributions and pull requests are currently closed for this project. If you wish to contribute or collaborate, please contact the maintainer first via email at [ajhmdni02@gmail.com](mailto:ajhmdni02@gmail.com).

---


## Credits & Acknowledgements

### Datasets
- **CAS(ME)^2**: Chinese Academy of Sciences Micro-expression and Macro-expression database.
- **SAMM**: Spontaneous Actions and Micro-Movement database.

### Core Libraries
- [MediaPipe](https://github.com/google/mediapipe) for face mesh tracking.
- [PyTorch](https://pytorch.org/) & [CuPy](https://cupy.dev/) for deep learning and GPU computations.
- [FastAPI](https://fastapi.tiangolo.com/) & [aiortc](https://github.com/aiortc/aiortc) for WebRTC and async APIs.
- [MinIO](https://min.io/) for S3 object storage.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](file:///home/inadio/skripkir/pulse-live/LICENSE) for details.

