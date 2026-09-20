"""Post-install patch: fix mediapipe <-> OpenCV ABI mismatch.

mediapipe 0.10.15 (custom wheel) links against libopencv_*.so.414 (OpenCV 4.14),
but the project ships OpenCV 4.15 (.so.415).  This script:

1. Patches libmediapipe.so NEEDED entries: .so.414 -> .so.415 (via patchelf)
2. Writes sitecustomize.py to pre-load .so.415 with RTLD_GLOBAL at Python startup

Idempotent -- safe to run multiple times.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

NEEDED_LIBS = [
    "libopencv_core",
    "libopencv_calib3d",
    "libopencv_features2d",
    "libopencv_highgui",
    "libopencv_imgcodecs",
    "libopencv_imgproc",
    "libopencv_video",
    "libopencv_videoio",
]


def _find_patchelf() -> str:
    venv_bin = Path(sys.executable).parent
    candidate = venv_bin / "patchelf"
    if candidate.is_file():
        return str(candidate)
    return "patchelf"


def _find_mediapipe_lib() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("mediapipe")
    if spec is None or spec.origin is None:
        raise SystemExit("mediapipe package not found in current environment")
    mp_init = Path(spec.origin)
    return mp_init.parent / "tasks" / "c" / "libmediapipe.so"


def _site_packages() -> Path:
    return Path(sysconfig.get_path("purelib"))


def _find_cv_libs() -> list[Path]:
    candidates: list[Path] = []

    cache_base = Path.home() / ".cache" / "uv" / "archive-v0"
    if cache_base.is_dir():
        for d in cache_base.iterdir():
            libs = d / "cv2" / "libs"
            if libs.is_dir():
                candidates.append(libs)

    sp = _site_packages()
    for name in ("opencv_contrib_python.libs", "opencv_contrib_python_headless.libs"):
        p = sp / name
        if p.is_dir():
            candidates.append(p)

    return candidates


def _has_soname(path: Path, soname: str) -> bool:
    try:
        out = subprocess.check_output(
            ["readelf", "-d", str(path)], text=True, stderr=subprocess.DEVNULL
        )
        return f"Shared library: [{soname}]" in out
    except Exception:
        return False


def _patch_mediapipe(patchelf: str, lib_path: Path) -> bool:
    bak = lib_path.with_suffix(".so.bak")
    if not bak.exists():
        bak.write_bytes(lib_path.read_bytes())

    needs_patch = any(
        _has_soname(lib_path, f"{lib}.so.414") for lib in NEEDED_LIBS
    )
    if not needs_patch:
        print(f"  [skip] {lib_path.name} already patched")
        return False

    for lib in NEEDED_LIBS:
        subprocess.run(
            [patchelf, "--replace-needed", f"{lib}.so.414", f"{lib}.so.415", str(lib_path)],
            check=True,
        )
    print(f"  [done] patched {lib_path.name}")
    return True


def _write_sitecustomize(cv_lib_dirs: list[Path]) -> None:
    sp = _site_packages()
    target = sp / "sitecustomize.py"

    dir_literals = ", ".join(f'"{d}"' for d in cv_lib_dirs)
    content = f"""\
import ctypes
import glob as _glob
import os

_CV_LIB_DIRS = [{dir_literals}]

for _d in _CV_LIB_DIRS:
    if not os.path.isdir(_d):
        continue
    for _lib in sorted(_glob.glob(os.path.join(_d, "libopencv_*.so.415"))):
        try:
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
"""
    target.write_text(content)
    print(f"  [done] wrote {target}")


def main() -> None:
    print("patch_mediapipe: checking mediapipe <-> OpenCV ABI ...")

    patchelf = _find_patchelf()
    lib_path = _find_mediapipe_lib()
    if not lib_path.exists():
        raise SystemExit(f"libmediapipe.so not found at {lib_path}")

    _patch_mediapipe(patchelf, lib_path)

    cv_libs = _find_cv_libs()
    if not cv_libs:
        print("  [warn] no opencv .so.415 directories found, skipping sitecustomize")
        return
    _write_sitecustomize(cv_libs)

    print("patch_mediapipe: all done.")


if __name__ == "__main__":
    main()
