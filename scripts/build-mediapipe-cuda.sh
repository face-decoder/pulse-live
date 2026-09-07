#!/usr/bin/env bash

# =========================================================
# MediaPipe build, pinned to the version this repo consumes
# (pyproject.toml `mediapipe==0.10.15`), packaged into the wheel
# at packages/mediapipe-<ver>-<tag>-<tag>-linux_x86_64.whl.
#
# Root cause this replaces: the checked-in wheel was built with
# `setup.py bdist_wheel --link-opencv`, which makes MediaPipe's
# libmediapipe.so dynamically link the *host's* system OpenCV and
# (through it) the host's system libjpeg. That happened to be
# libjpeg.so.8 (Debian/Ubuntu libjpeg-turbo SONAME) on whatever
# machine built the checked-in wheel. Fedora's libjpeg-turbo is
# libjpeg.so.62 — no libjpeg.so.8 exists here, so the import fails.
#
# Fix: drop --link-opencv. Bazel then fetches/builds its own pinned
# OpenCV + libjpeg-turbo from source (per MediaPipe's WORKSPACE) and
# links them statically into libmediapipe.so, so the resulting wheel
# has no dependency on the host's OpenCV or libjpeg at all — same
# self-contained idea as make_opencv_wheel.py's cv2.libs bundling,
# achieved here by *not* depending on host libs in the first place.
#
# Cross-platform system deps: Ubuntu/Debian, Fedora/RHEL/CentOS.
# =========================================================

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MEDIAPIPE_VERSION="${MEDIAPIPE_VERSION:-v0.10.15}"
DEV_DIR="${MEDIAPIPE_BUILD_DIR:-$HOME/dev-build}"
MEDIAPIPE_SRC="$DEV_DIR/mediapipe"

section() { printf "\n=========================================================\n%s\n=========================================================\n\n" "$1"; }
info() { printf "[INFO] %s\n" "$1"; }
error() { printf "[ERROR] %s\n" "$1"; }

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
    else
        OS_ID="unknown"
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
        *)
            error "Unsupported OS: $OS_ID. Install build dependencies manually."
            exit 1
            ;;
    esac
}

section "Installing system build dependencies"

case "$OS_ID" in
    ubuntu|debian|linuxmint|pop)
        pkg_install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev \
            build-essential unzip curl git protobuf-compiler
        ;;
    fedora|rhel|centos|rocky|almalinux)
        pkg_install mesa-libEGL-devel mesa-libGLES-devel \
            gcc gcc-c++ unzip curl git protobuf-compiler protobuf-devel
        ;;
esac

section "Resolving bazelisk"

BAZELISK_BIN="$DEV_DIR/bin/bazelisk"
if [ ! -x "$BAZELISK_BIN" ]; then
    mkdir -p "$DEV_DIR/bin"
    curl -fL -o "$BAZELISK_BIN" \
        "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
    chmod +x "$BAZELISK_BIN"
fi
# setup.py shells out to a literal `bazel` binary — bazelisk resolves the
# right version and proxies to it, so expose it under that name too.
ln -sf bazelisk "$DEV_DIR/bin/bazel"
export PATH="$DEV_DIR/bin:$PATH"
info "bazelisk: $(bazelisk version 2>&1 | head -1)"

section "Resolving build Python interpreter"

if ! command -v uv &> /dev/null; then
    error "uv not found on PATH — required to resolve the repo's pinned Python."
    exit 1
fi

# Isolated venv for the mediapipe build itself, kept separate from the
# repo's .venv so build-only deps (numpy/opencv from requirements.txt)
# never clash with the repo's pinned versions (numpy==1.26.4, the CUDA
# opencv-contrib-python wheel). Same Python 3.12 -> same cp312 wheel tag.
BUILD_VENV="$DEV_DIR/build-venv"
# Read directly instead of `uv run python -c ...`: `uv run` auto-syncs the
# whole project first, and if the repo's lockfile is out of date that can
# trigger a large unrelated dependency download just to read a version number.
REPO_PY_VERSION="$(cat "$REPO_ROOT/.python-version")"
[ -d "$BUILD_VENV" ] || uv venv --python "$REPO_PY_VERSION" "$BUILD_VENV" --quiet
PYTHON_EXEC="$BUILD_VENV/bin/python"
PYTHON_TAG="$("$PYTHON_EXEC" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')"

info "Python executable : $PYTHON_EXEC"
info "Python tag        : $PYTHON_TAG"

section "Fetching MediaPipe $MEDIAPIPE_VERSION source"

mkdir -p "$DEV_DIR"
if [ -d "$MEDIAPIPE_SRC/.git" ]; then
    git -C "$MEDIAPIPE_SRC" fetch --depth 1 origin tag "$MEDIAPIPE_VERSION"
    git -C "$MEDIAPIPE_SRC" checkout "$MEDIAPIPE_VERSION"
    # Re-running against an already-checked-out tag no-ops the checkout above,
    # so this script's own sed patches (below) would otherwise accumulate
    # across reruns instead of applying to a clean tree each time.
    git -C "$MEDIAPIPE_SRC" reset --hard "$MEDIAPIPE_VERSION"
    git -C "$MEDIAPIPE_SRC" clean -fdx
else
    git clone --branch "$MEDIAPIPE_VERSION" --depth 1 \
        https://github.com/google/mediapipe.git "$MEDIAPIPE_SRC"
fi

cd "$MEDIAPIPE_SRC"

# jax/jaxlib are declared in requirements.txt but mediapipe's Python task API
# (what this repo actually imports/uses) never imports them — they're only
# used by optional experimental codepaths. Pulling them in as a hard runtime
# dependency means uv resolving a multi-GB unneeded download. Strip them so
# the wheel's metadata (setup.py embeds requirements.txt verbatim as
# install_requires) matches what's actually needed.
sed -i '/^jax$/d;/^jaxlib$/d' requirements.txt

uv pip install -q --python "$PYTHON_EXEC" setuptools wheel
uv pip install -q --python "$PYTHON_EXEC" -r "$MEDIAPIPE_SRC/requirements.txt"

# setup.py hardcodes __version__ = 'dev', which modern setuptools/packaging
# rejects as non-PEP440 (InvalidVersion). Stamp the pinned version instead.
sed -i "s/^__version__ = 'dev'$/__version__ = '${MEDIAPIPE_VERSION#v}'/" setup.py

# third_party/BUILD hardcodes the OpenCV-from-source link flags for an
# Ubuntu-era OpenEXR 2.x (-lIlmImf -lHalf -lIex -lIlmThread -lImath) and old
# ffmpeg's -lavresample. Fedora 44 ships OpenEXR 3.x (those SONAMEs no longer
# exist, merged into libOpenEXR/libImath) and dropped libavresample in favor
# of libswresample years ago — this is the exact same "hardcoded distro-era
# lib names" problem the checked-in wheel had, just one layer deeper (inside
# mediapipe's own vendored OpenCV build instead of the final .so). Disable
# OpenEXR support (unneeded: only affects .exr image codec support). Also
# disable OpenCV's ffmpeg videoio backend: OpenCV's bundled cap_ffmpeg_impl.hpp
# here still uses pre-2013 `CODEC_ID_*` names (renamed to `AV_CODEC_ID_*`
# many ffmpeg releases ago) and won't compile against ffmpeg 8.1's headers —
# unneeded anyway, mediapipe/FaceLandmarker doesn't read video files through
# OpenCV's videoio.
#
# Also: CMake's GNUInstallDirs defaults to lib64/ for the static libs on a
# 64-bit Fedora/RHEL system (Debian/Ubuntu default to plain lib/), but the
# bazel cmake() rule hardcodes out_static_libs under .../lib/libopencv_*.a —
# another distro-convention mismatch, one layer deeper. Force lib/.
sed -i \
    -e '/"WITH_JASPER": "OFF",/a\        "WITH_OPENEXR": "OFF",\n        "WITH_FFMPEG": "OFF",\n        "CMAKE_INSTALL_LIBDIR": "lib",\n        "WITH_IPP": "OFF",' \
    -e '/"-lImath",/d' -e '/"-lIlmImf",/d' -e '/"-lIex",/d' \
    -e '/"-lHalf",/d' -e '/"-lIlmThread",/d' \
    -e '/"-lavcodec",/d' -e '/"-lavformat",/d' -e '/"-lavutil",/d' \
    -e '/"-lswscale",/d' -e '/"-lavresample",/d' \
    third_party/BUILD

section "Building MediaPipe wheel (statically-linked OpenCV/libjpeg)"

# docs/BUILD_MEDIAPIPE_GPU.md's `--bazel-flags="--config=cuda ..."` doesn't
# apply here: this setup.py has no --bazel-flags option, and this version of
# mediapipe's .bazelrc has no `cuda` config at all — its "GPU" build here is
# the EGL/GLES mobile delegate, not a CUDA one (that doc conflated this with
# TensorFlow's own build). Real option is --link-opencv (default off, which
# is what we want): bazel then builds+links OpenCV/libjpeg from source
# instead of the host's, which is what made the old wheel non-portable.
"$PYTHON_EXEC" setup.py bdist_wheel

WHEEL_FILE="$(find dist -maxdepth 1 -name "mediapipe-*-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl" | head -1)"
if [ -z "$WHEEL_FILE" ]; then
    error "No wheel produced matching tag $PYTHON_TAG in $MEDIAPIPE_SRC/dist"
    exit 1
fi

mkdir -p "$REPO_ROOT/packages"
cp "$WHEEL_FILE" "$REPO_ROOT/packages/"
info "Wheel copied to: $REPO_ROOT/packages/$(basename "$WHEEL_FILE")"

info "Next: uv lock --refresh-package mediapipe && uv sync --reinstall-package mediapipe"
