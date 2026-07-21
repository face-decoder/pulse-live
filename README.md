# Pulse Live

Facial micro-expression detection and real-time analysis system using computer vision, optical flow, and deep learning techniques.

---

## Project Overview

**Pulse Live** is an end-to-end computer vision and deep learning platform designed for real-time and offline detection, spotting, and classification of facial micro-expressions. Micro-expressions are subtle, fleeting involuntary facial movements that reveal underlying emotional states and psychological responses.

The system incorporates a multi-stage analysis pipeline:
- **Face Mesh & Landmark Alignment**: Real-time tracking and alignment using MediaPipe FaceMesh to isolate key Regions of Interest (ROIs) such as eyebrows, eyes, and mouth.
- **Phase-Based Apex Spotting**: Utilizing Riesz Pyramids and motion magnitude analysis to spot onset, apex, and offset frames of micro-movements.
- **Dense Optical Flow Extraction**: Extracting spatial-temporal motion dynamics via TV-L1 optical flow algorithms (accelerated with CUDA / CuPy).
- **Deep Learning Classification**: Running sequence modeling architectures (CNN-Transformer, CNN-BiLSTM, TCN, Spatio-Temporal networks) with Test-Time Augmentation (TTA) to predict behavioral and emotional states.
- **Real-Time Telemetry & Offloading**: WebRTC (`aiortc`) and WebSocket streaming channels for live feedback, coupled with MinIO S3 object storage for artifact persistence.

---

## Features

- **Real-Time Video Streaming**: Low-latency video stream ingestion over WebRTC (`aiortc`) and binary WebSocket protocols with real-time bounding box and analytics delivery.
- **Offline Video Processing**: REST API endpoints for uploading pre-recorded video files with frame-by-frame feature extraction, micro-expression spotting, and exportable results.
- **Phase-Based Apex Spotting**: Advanced Riesz Pyramid phase analysis for precise detection of micro-expression peak frames across short temporal windows.
- **Face Mesh & ROI Extraction**: MediaPipe FaceMesh integration for head pose alignment, facial landmark visualizers, and targeted ROI cropping (eyebrows, eyes, nose, mouth).
- **CUDA-Accelerated Optical Flow**: High-performance TV-L1 optical flow calculation optimized using CuPy and GPU kernels.
- **Multi-Architecture Deep Learning Suite**:
  - Transformer
  - Bi-LSTM (with Attention / Multi-Head Attention)
  - LSTM + MLP
  - Temporal Convolutional Networks (TCN)
  - Spatio-Temporal 3D Neural Networks
- **Test-Time Augmentation (TTA)**: Configurable multi-pass TTA during inference for robust model predictions.
- **MinIO S3 Integration**: Automated object storage management for session recordings, landmark metadata, and generated optical flow artifacts.
- **Telemetry & Historical Analytics**: Comprehensive API for retrieving real-time inference history, frame-level metrics, and system log streams.

---

## Tech Stack

| Category | Technology |
| :--- | :--- |
| **Programming Language** | Python 3.12 |
| **Deep Learning & Math** | PyTorch (CUDA 12.8), Torchvision, SciPy, NumPy, Scikit-Learn |
| **GPU Acceleration** | CuPy (CUDA 12.x) |
| **Computer Vision** | OpenCV, MediaPipe (Custom Linux GPU Wheel), PyAV (`av`) |
| **Web Framework & API** | FastAPI, Uvicorn, WebSockets, `aiortc` (WebRTC) |
| **Object Storage** | MinIO (S3-Compatible Storage) |
| **Containerization** | Docker, Docker Compose |
| **Package & Env Manager** | `uv` |

---

## Installation

### Prerequisites

- **Operating System**: Linux (Ubuntu 22.04+ recommended)
- **Python**: Python 3.12
- **GPU Driver & CUDA**: NVIDIA GPU Driver supporting CUDA 12.x / CUDA 12.8
- **Tooling**: [`uv`](https://github.com/astral-sh/uv) package manager, Docker & Docker Compose, `make`

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/pulse-live.git
   cd pulse-live
   ```

2. **Configure Environment Variables**
   Create or update the `.env` configuration file:
   ```bash
   cp .env.example .env  # or edit .env directly
   ```

3. **Install Dependencies**
   Use `uv` to synchronize dependencies and environment:
   ```bash
   make sync-deps
   # or directly with uv:
   uv sync
   ```

4. **Start Infrastructure Services**
   Launch MinIO S3 object storage container:
   ```bash
   make infra
   ```

---

## Commands

The project includes a `Makefile` to streamline common operations.

| Command | Description |
| :--- | :--- |
| `make run` | Run the FastAPI development server with Uvicorn (`uv run python main.py`) |
| `make dev` | Development shortcut to start the server with auto-reload |
| `make infra` | Start Docker containers (MinIO object storage) in detached mode |
| `make infra-down` | Stop and remove running Docker containers |
| `make infra-logs` | Tail real-time logs from Docker container infrastructure |
| `make sync-deps` | Run `./scripts/sync-deps.sh` to synchronize environment dependencies |
| `make clean` | Clean up temporary files and python caches (`__pycache__`, `.tmp`) |

### Direct Script & Execution Commands

- **Run Web Application Server**:
  ```bash
  uv run python main.py
  ```
- **Execute Micro-Expression Spotting Benchmarks**:
  ```bash
  uv run python cas_me_2_spotting.py
  uv run python samm_spotting.py
  ```
- **Generate Apex Visualizations**:
  ```bash
  uv run python generate_apex_visual.py
  ```

---

## Project Structure

```
pulse-live/
├── .env                          # Environment configuration & model settings
├── docker-compose.yml            # Docker infrastructure configuration (MinIO)
├── Makefile                      # Makefile for running and managing the app
├── pyproject.toml                # UV dependency manifest and PyPI index specs
├── main.py                       # FastAPI application entrypoint & lifespan manager
├── real-time.log                 # Server runtime log file
├── docs/                         # Detailed project documentation & API contracts
│   ├── API_CONTRACT.md           # WebRTC & WebSocket signaling schema spec
│   ├── BUILD_MEDIAPIPE_GPU.md    # Instructions for building MediaPipe with GPU support
│   ├── LOGS_API.md               # Telemetry and logging API documentation
│   └── VIDEO_UPLOAD.md           # Video upload processing contract
├── packages/                     # Custom pre-compiled binary packages (.whl)
│   └── mediapipe-0.10.15-*.whl   # Custom MediaPipe Linux GPU wheel
├── scripts/                      # Utility and setup scripts
│   ├── generate_cas_me_2_cache.py
│   └── sync-deps.sh
├── combinations-notebooks/       # Model checkpoints and trained weights
├── notebooks/                    # Jupyter notebooks for data analysis & experiments
└── src/                          # Core application source code
    ├── api/                      # FastAPI routers (WebRTC, WebSocket, Video, History, Logs)
    ├── apex/                     # Micro-expression apex frame spotters & phase analysis
    ├── dataset/                  # Dataset loaders, feature augmentations, transforms
    ├── datasource/               # Window and hybrid sequence data extractors
    ├── evaluator/                # Metrics and performance evaluation utilities
    ├── face/                     # MediaPipe face mesh, landmark tracking, ROI aligners
    ├── models/                   # Deep learning models & inference registry
    │   ├── inferencer/           # Model inferencer implementations (CNN-Transformer, TCN, etc.)
    │   └── modules/              # PyTorch neural network layer components
    ├── optical_flow/             # CUDA-accelerated TV-L1 optical flow module
    ├── storage/                  # MinIO S3 object storage client & persistence
    ├── utils/                    # Common helpers and signal processing utilities
    └── video/                    # Video decoding, frame extraction, FPS handlers
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](file:///home/inadio/skripkir/pulse-live/LICENSE) file for details.
