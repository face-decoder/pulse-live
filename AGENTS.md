# AGENTS.md — Workspace Directives & Repository Architecture

> **CRITICAL INSTRUCTION FOR ALL AI AGENTS**:
> You MUST read and strictly adhere to these instructions before executing ANY task or responding in this repository. These rules are persistent, mandatory, and take precedence over default behaviors.

---

## 1. Mandatory Execution Rules (`/ponytail`)

Execute all code tasks using the **simplest, shortest, most minimal solution** that actually works. Think like an experienced developer who values simplicity, maintainability, and efficiency.

### The Ladder of Simplicity
Climb this ladder and stop at the first rung that works:
1. **Does this need to exist at all? (YAGNI)**: If the requirement is speculative, skip it.
2. **Already in this codebase?**: Reuse existing helpers, utilities, types, and modules (`src/apex/`, `src/face/`, `src/api/`, etc.) before writing new ones. Never re-implement what already exists.
3. **Python Standard Library first**: Reach for built-in modules (`os`, `sys`, `pathlib`, `json`, `math`, `asyncio`) before writing custom helpers.
4. **Existing installed dependencies**: Use installed packages (`numpy`, `scipy`, `torch`, `cv2`, `cupy`, `fastapi`, `aiortc`). NEVER add new dependencies for what a few lines of code can achieve.
5. **Can it be one line?**: Make it one line.
6. **Minimum working code**: Write only what is strictly required to satisfy the user request.

### Engineering Constraints
- **No Unrequested Abstractions**: No interfaces with a single implementation, no wrapper classes around one call, no factory for a single product, no unused configuration variables.
- **No Scaffolding / Boilerplate**: Do not write code for hypothetical future requirements.
- **Deletion over Addition**: Prefer simplifying, refactoring down, and deleting dead code over adding layers.
- **Root-Cause Fixes**: Fix bugs at their root in shared modules rather than patching symptoms at multiple call sites.

---

## 2. Mandatory Response Style Rules (`/caveman`)

Format all conversational and explanatory responses in **terse, smart caveman style**:
- **Drop filler words**: Omit pleasantries, conversational fluff, introductory greetings, and decorative transitions.
- **Drop unnecessary articles**: Drop non-essential articles (`a`, `an`, `the`) where meaning remains unambiguous. Sentence fragments are allowed.
- **Preserve Technical Precision**: Technical terms, variable names, function names, metrics, and file paths MUST remain exact and properly formatted.
- **Code Integrity**: Code blocks, diffs, configuration files, and terminal commands must remain complete, fully functional, and unaltered by caveman compression.
- **Response Pattern**: `[thing] [action] [reason]. [next step].`

---

## 3. Repository Architecture Summary (Based on `master` Branch)

The `pulse-live` repository is a high-performance, real-time facial micro-expression spotting (RMES) and emotion classification system designed for WebRTC/WebSocket streaming and offline batch video analytics.

### Core Technology Stack
- **Language & Runtime**: Python 3.12 (managed via `uv`)
- **Deep Learning**: PyTorch (CUDA 12.8), Torchvision, Scikit-Learn, SciPy, NumPy
- **GPU Acceleration**: CuPy, OpenCV CUDA 4.15.0, Custom MediaPipe GPU wheel (`packages/mediapipe-0.10.15-cp312-cp312-linux_x86_64.whl`)
- **Web & Streaming**: FastAPI, Uvicorn, WebSockets, `aiortc` (WebRTC)
- **Object Storage**: MinIO (S3-compatible bucket storage for video frames and telemetry metadata)
- **Infrastructure**: Docker, Docker Compose, GNU Makefile

---

### Module Hierarchy & Architecture Breakdown

```
src/
├── apex/                  # Riesz Pyramid Phase-Based Apex Spotting
│   └── modules/
│       ├── apex_phase.py              # Spatial Riesz transform & monogenic phase differences
│       ├── apex_phase_spotter.py      # Core phase spotter base class
│       ├── apex_phase_spotter_fullface.py # Fullface phase spotting implementation
│       ├── apex_phase_spotter_roi.py  # FACS ROI-based phase spotter
│       ├── apex_smoother.py           # FIR lowpass and moving average score smoothers
│       ├── apex_spotter.py            # Optical flow & baseline apex spotters
│       └── apex_phase_visualizer.py   # Quiver and phase motion plotting tools
│
├── face/                  # Facial Analysis, Alignment & Tracking
│   ├── modules/
│   │   ├── face_landmark.py           # MediaPipe FaceMesh (GPU/CPU) 68/478 landmark detection
│   │   ├── face_aligner.py            # Affine transformation & pose normalization (224x224)
│   │   ├── face_roi.py                # FACS Action Unit ROI extraction (eyebrows, eyes, mouth)
│   │   └── face_landmark_visualizer.py# Landmark overlay & mesh debug visualizers
│   └── tasks/
│       └── face_landmarker.task       # MediaPipe binary task model bundle
│
├── optical_flow/          # High-Speed Motion Vector Estimation
│   └── modules/
│       └── tvl1.py                    # Dual TV-L1 optical flow via CUDA/CuPy kernels
│
├── models/                # Deep Neural Networks & Inferencer Suite
│   ├── modules/                       # Model definitions: CNN-BiLSTM, CNN-Transformer,
│   │                                  # CNN-BiLSTM-Attention, CNN-BiLSTM-MHA, TCN,
│   │                                  # Spatio-Temporal 3D CNN, Positional Encoding
│   └── inferencer/
│       ├── _factory.py                # Inferencer factory instantiator
│       ├── registry.py                # Model registry and metadata mapping
│       ├── base.py                    # Abstract base inferencer class
│       └── [model]_inferencer.py      # Concrete inferencers with Test-Time Augmentation (TTA)
│
├── api/                   # WebRTC / WebSocket Streaming & REST Endpoints
│   ├── webrtc.py                      # WebRTC peer connection handler (`/api/webrtc/offer`)
│   ├── websocket.py                   # Real-time telemetry WebSocket (`/api/ws/telemetry`)
│   ├── video_process.py               # Video file upload & processing (`/api/video/upload`)
│   ├── stream_processor.py            # Real-time frame queue, inference pipeline & dispatch
│   ├── connection_manager.py          # WebSocket client subscription manager
│   ├── window_buffers.py              # Sliding window frame ring-buffers
│   └── session_state.py               # Client session state & telemetry cache
│
├── dataset/               # Academic Dataset Pipelines & Data Augmentation
│   ├── modules/
│   │   ├── anxiety_dataset_base.py    # Base dataset loader class
│   │   ├── flow_roi_dataset.py        # FACS ROI optical flow / phase dataset loader
│   │   ├── flow_fullface_dataset.py   # Fullface motion dataset loader
│   │   ├── channel_zscore.py          # Channel-wise Z-score normalization
│   │   ├── temporal_transforms.py     # Frame interpolation & temporal resampling
│   │   └── augment_flow.py            # Motion field spatial augmentations
│   └── constants/                     # Dataset constants for CAS(ME)2 and SAMM
│
├── evaluator/             # Model Benchmarking & Metric Calculations
│   └── modules/
│       ├── feature_extractor.py       # Intermediate feature map extraction
│       └── metric_evaluator.py        # LOSO evaluation, IoU >= 0.5 thresholding, F1 scores
│
├── storage/               # Object Storage Client
│   └── modules/
│       └── minio_client.py            # MinIO client for video/frame and landmark persistence
│
├── video/                 # Frame Extraction & Video Decoders
│   └── modules/
│       ├── video.py                   # PyAV / OpenCV frame stream decoder
│       └── lazy_video.py              # On-demand lazy frame loader
│
├── plotter/               # Evaluation Visualization
│   └── modules/                       # Confusion matrix, ROC curve, t-SNE, and training history plots
│
└── utils/                 # System Utilities
    └── env.py                         # Environment variables and configuration loader
```

---

### Key Data Flow & Pipelines (`master`)

1. **Live Stream Track**:
   `WebRTC Stream` -> `MediaPipe Landmarking` -> `Face Alignment (224x224)` -> `FACS ROI Cropping`
2. **Motion & Apex Spotting**:
   `Aligned Face ROIs` -> `Riesz Pyramid Phase Difference` -> `FIR Lowpass Filtering` -> `Moving Average Apex Spotter`
3. **Inference & Telemetry**:
   `Extracted Motion Tensors` -> `PyTorch Inferencer (CNN-BiLSTM / Transformer)` -> `WebSocket Telemetry Dispatch` -> `MinIO Storage`
