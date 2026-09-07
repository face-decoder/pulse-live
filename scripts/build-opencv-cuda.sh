#!/usr/bin/env bash

# =========================================================
# OpenCV + CUDA build, pinned to a real upstream release and
# packaged into the wheel this repo consumes via
# pyproject.toml's `cuda` extra.
#
# Cross-platform system deps: Ubuntu/Debian, Fedora/RHEL/CentOS, Arch, openSUSE.
# Linux only (CUDA doesn't run on macOS; Windows not supported here).
#
# Override via env vars:
#   OPENCV_VERSION      (default: 5.0.0)
#   OPENCV_BUILD_DIR    (default: $HOME/dev-build)
#   OPENCV_INSTALL_PREFIX (default: $OPENCV_BUILD_DIR/opencv_install)
#   CUDA_ARCH_BIN       (default: auto-detected from nvidia-smi)
# =========================================================

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPENCV_VERSION="${OPENCV_VERSION:-5.0.0}"
DEV_DIR="${OPENCV_BUILD_DIR:-$HOME/dev-build}"

OPENCV_SRC="$DEV_DIR/opencv"
OPENCV_CONTRIB="$DEV_DIR/opencv_contrib"
BUILD_DIR="$DEV_DIR/opencv_build"
INSTALL_PREFIX="${OPENCV_INSTALL_PREFIX:-$DEV_DIR/opencv_install}"

# =========================================================
# UI HELPERS
# =========================================================

section() { printf "\n=========================================================\n%s\n=========================================================\n\n" "$1"; }
info() { printf "[INFO] %s\n" "$1"; }
error() { printf "[ERROR] %s\n" "$1"; }
warn() { printf "[WARN] %s\n" "$1"; }

# =========================================================
# OS DETECTION
# =========================================================

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
        OS_LIKE="${ID_LIKE:-$ID}"
    elif command -v lsb_release &> /dev/null; then
        OS_ID=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
        OS_VERSION=$(lsb_release -sr)
        OS_LIKE="$OS_ID"
    else
        OS_ID="unknown"
        OS_VERSION="unknown"
        OS_LIKE="unknown"
    fi
}

detect_os

pkg_install() {
    case "$OS_ID" in
        ubuntu|debian|linuxmint|pop)
            sudo apt-get update -qq
            sudo apt-get install -y -qq "$@"
            ;;
        fedora|rhel|centos|rocky|almalinux)
            sudo dnf install -y "$@"
            ;;
        arch|manjaro|endeavouros)
            sudo pacman -Sy --noconfirm "$@"
            ;;
        opensuse*|sles)
            sudo zypper install -y "$@"
            ;;
        *)
            error "Unsupported OS: $OS_ID. Install build dependencies manually."
            exit 1
            ;;
    esac
}

# =========================================================
# CUDA PATH / ARCH DETECTION
# =========================================================

detect_cuda_path() {
    local paths=("/usr/local/cuda" "/usr/local/cuda-13.3" "/usr/local/cuda-13.0" "/usr/local/cuda-12.8" "/usr/local/cuda-12.6" "/usr/local/cuda-12.5" "/usr/local/cuda-12.4" "/usr/local/cuda-12.3" "/usr/local/cuda-12.2" "/usr/local/cuda-12.1" "/usr/local/cuda-12.0")

    for path in "${paths[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
            echo "$path"
            return
        fi
    done

    if command -v nvcc &> /dev/null; then
        local nvcc_path
        nvcc_path=$(command -v nvcc)
        echo "$(dirname "$(dirname "$nvcc_path")")"
        return
    fi

    echo ""
}

detect_cuda_arch() {
    if [ -n "${CUDA_ARCH_BIN:-}" ]; then
        echo "$CUDA_ARCH_BIN"
        return
    fi

    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found, using default CUDA arch" >&2
        echo "7.5"
        return
    fi

    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)

    if [ -z "$gpu_name" ]; then
        warn "No NVIDIA GPU detected, using default CUDA arch" >&2
        echo "7.5"
        return
    fi

    info "Detected GPU: $gpu_name" >&2

    case "$gpu_name" in
        *RTX\ 50*|*Blackwell*) echo "12.0" ;;
        *RTX\ 40*|*Ada*) echo "8.9" ;;
        *RTX\ 30*|*Ampere*) echo "8.6" ;;
        *RTX\ 20*|*TITAN\ RTX*|*Turing*) echo "7.5" ;;
        *GTX\ 16*) echo "7.5" ;;
        *GTX\ 10*|*Pascal*) echo "6.1" ;;
        *Tesla*|*A100*|*H100*|*H200*|*B100*|*B200*) echo "8.0" ;;
        *V100*|*Volta*) echo "7.0" ;;
        *Quadro*) echo "7.5" ;;
        *)
            warn "Unknown GPU model '$gpu_name', using compute capability 7.5" >&2
            echo "7.5"
            ;;
    esac
}

CUDA_PATH=$(detect_cuda_path)

if [ -z "$CUDA_PATH" ] || [ ! -d "$CUDA_PATH" ]; then
    error "CUDA toolkit not found. Install it first: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_ARCH=$(detect_cuda_arch)

# =========================================================
# REPO'S PYTHON (must match the cp312 wheel the rest of the
# project consumes via uv, not whatever python3 is on PATH)
# =========================================================

section "Resolving repo Python interpreter"

if ! command -v uv &> /dev/null; then
    error "uv not found on PATH — required to resolve the repo's pinned Python."
    exit 1
fi

( cd "$REPO_ROOT" && uv sync --quiet ) # ensure .venv exists before we query it
PYTHON_EXEC="$(cd "$REPO_ROOT" && uv run python -c 'import sys; print(sys.executable)')"
PYTHON_TAG="$(cd "$REPO_ROOT" && uv run python -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"

info "Python executable : $PYTHON_EXEC"
info "Python tag        : $PYTHON_TAG"

# =========================================================
# HEADER
# =========================================================

section "OpenCV + CUDA build ($OPENCV_VERSION)"

info "Operating System   : $OS_ID $OS_VERSION ($OS_LIKE)"
info "OpenCV Version     : $OPENCV_VERSION"
info "OpenCV Source      : $OPENCV_SRC"
info "OpenCV Contrib     : $OPENCV_CONTRIB"
info "Build Directory    : $BUILD_DIR"
info "Install Prefix     : $INSTALL_PREFIX"
info "CUDA Path          : $CUDA_PATH"
info "CUDA Architecture  : $CUDA_ARCH"

# =========================================================
# SYSTEM BUILD DEPENDENCIES
# =========================================================

if command -v gcc &> /dev/null && command -v cmake &> /dev/null && command -v ninja &> /dev/null && command -v pkg-config &> /dev/null; then
    section "System build dependencies already present, skipping install"
else

section "Installing system build dependencies"

case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
        pkg_install build-essential cmake ninja-build pkg-config \
            git wget curl unzip \
            libgtk-3-dev libjpeg-dev libpng-dev libtiff-dev \
            libavcodec-dev libavformat-dev libswscale-dev \
            libv4l-dev libxvidcore-dev libx264-dev \
            libatlas-base-dev gfortran libeigen3-dev libtbb-dev \
            libopenexr-dev libdc1394-dev \
            libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
            libgl1-mesa-dev libglu1-mesa-dev 2>/dev/null || true
        ;;
    fedora)
        pkg_install gcc gcc-c++ cmake ninja-build pkg-config \
            git wget curl unzip gtk3-devel \
            libjpeg-turbo-devel libpng-devel libtiff-devel ffmpeg-free-devel \
            libv4l-devel atlas-devel gcc-gfortran eigen3-devel tbb-devel \
            openexr-devel libdc1394-devel \
            gstreamer1-devel gstreamer1-plugins-base-devel \
            mesa-libGL-devel mesa-libGLU-devel
        ;;
    rhel|centos|rocky|almalinux)
        pkg_install gcc gcc-c++ cmake3 ninja-build pkgconfig \
            git wget curl unzip gtk3-devel \
            libjpeg-turbo-devel libpng-devel libtiff-devel ffmpeg-devel \
            libv4l-devel x264-devel atlas-devel gcc-gfortran eigen3-devel tbb-devel \
            openexr-devel gstreamer1-devel gstreamer1-plugins-base-devel \
            mesa-libGL-devel mesa-libGLU-devel
        ;;
    arch|manjaro|endeavouros)
        pkg_install base-devel cmake ninja pkg-config \
            git wget curl unzip gtk3 \
            libjpeg-turbo libpng libtiff ffmpeg libv4l2-tools x264 \
            openblas eigen tbb openexr libdc1394 gst-plugins-base mesa
        ;;
    opensuse*|sles)
        pkg_install gcc gcc-c++ cmake ninja pkg-config \
            git wget curl unzip gtk3-devel \
            libjpeg8-devel libpng16-devel libtiff-devel \
            ffmpeg-4-libavcodec-devel ffmpeg-4-libavformat-devel \
            libv4l-devel x264-devel libatlas-devel gcc-fortran eigen3-devel tbb-devel \
            openexr-devel gstreamer-plugins-base-devel Mesa-libGL-devel
        ;;
esac

fi

# =========================================================
# FETCH SOURCE, PINNED TO $OPENCV_VERSION
# =========================================================

section "Fetching OpenCV $OPENCV_VERSION source"

mkdir -p "$DEV_DIR"

fetch_pinned() {
    local url="$1" dir="$2"
    if [ -d "$dir/.git" ]; then
        git -C "$dir" fetch --depth 1 origin "tag" "$OPENCV_VERSION"
        git -C "$dir" checkout "$OPENCV_VERSION"
    else
        git clone --branch "$OPENCV_VERSION" --depth 1 "$url" "$dir"
    fi
}

fetch_pinned "https://github.com/opencv/opencv.git" "$OPENCV_SRC"
fetch_pinned "https://github.com/opencv/opencv_contrib.git" "$OPENCV_CONTRIB"

if [ ! -f "$OPENCV_SRC/CMakeLists.txt" ]; then
    error "Invalid OpenCV source directory: $OPENCV_SRC"
    exit 1
fi

# =========================================================
# CONFIGURE + BUILD + INSTALL
# =========================================================

section "Cleaning previous build"

rm -rf "$BUILD_DIR"
rm -rf "$INSTALL_PREFIX"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

section "Running CMake"

cmake -S "$OPENCV_SRC" -B . \
-G Ninja \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
-D OPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB/modules" \
-D WITH_CUDA=ON \
-D WITH_CUDNN=OFF \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH" \
-D CUDA_ARCH_BIN="$CUDA_ARCH" \
-D CUDA_NVCC_FLAGS="-allow-unsupported-compiler" \
-D CUDA_USE_STATIC_CUDA_RUNTIME=OFF \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D WITH_CUBLAS=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_V4L=ON \
-D WITH_TBB=ON \
-D WITH_OPENMP=ON \
-D WITH_EIGEN=ON \
-D WITH_IPP=ON \
-D WITH_OPENGL=ON \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D WITH_FREETYPE=OFF \
-D WITH_HARFBUZZ=OFF \
-D BUILD_opencv_python3=ON \
-D PYTHON3_EXECUTABLE="$PYTHON_EXEC" \
-D PYTHON3_LIBRARY="$($PYTHON_EXEC -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") + "/" + sysconfig.get_config_var("LDLIBRARY"))')" \
-D PYTHON3_INCLUDE_DIR="$($PYTHON_EXEC -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
-D PYTHON3_PACKAGES_PATH="$INSTALL_PREFIX/lib/python3" \
-D PYTHON3_NUMPY_INCLUDE_DIRS="$($PYTHON_EXEC -c 'import numpy; print(numpy.get_include())')" \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_DOCS=OFF \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_NVCUVID=OFF \
-D WITH_NVCUVENC=OFF \
2>&1 | tee cmake.log

section "Building OpenCV"

ninja 2>&1 | tee build.log

section "Installing OpenCV to $INSTALL_PREFIX"

ninja install 2>&1 | tee install.log

# =========================================================
# PACKAGE INTO WHEEL
# =========================================================

section "Packaging wheel"

"$PYTHON_EXEC" "$REPO_ROOT/scripts/make_opencv_wheel.py" \
    --python-packages-dir "$INSTALL_PREFIX/lib/python3" \
    --install-prefix "$INSTALL_PREFIX" \
    --version "$OPENCV_VERSION" \
    --python-tag "$PYTHON_TAG" \
    --output-dir "$REPO_ROOT/packages"

section "Verifying build"

LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:$CUDA_PATH/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$INSTALL_PREFIX/lib/python3:${PYTHONPATH:-}" \
"$PYTHON_EXEC" -c '
import cv2
print("OpenCV version:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
'

section "OpenCV + CUDA build complete"

info "Wheel written to: $REPO_ROOT/packages/"
info "Next: make sync-deps-cuda && make run-cuda"
