#!/usr/bin/env python3
"""
Package a cmake-built OpenCV Python module into an installable wheel.

Ships the wheel as `opencv-contrib-python` (not a made-up `opencv-cuda` name):
mediapipe already depends on real `opencv-contrib-python` from PyPI to get its
`cv2` import. Two separate packages both trying to own the top-level `cv2`
namespace is what corrupted the previous hand-made wheel install. Overriding
the *same* dependency name via `tool.uv.sources` (marker-gated on the `cuda`
extra) means there is exactly one `cv2` provider at a time — no collision.
"""

from __future__ import annotations

import argparse
import hashlib
import base64
import re
import zipfile
from pathlib import Path

# Only the SONAME-form file (e.g. libopencv_core.so.500) is needed at runtime —
# that's what the cv2 extension's DT_NEEDED entries reference. The bare `.so`
# (compile-time devlink) and the fully-versioned `.so.5.0.0` real file are
# redundant copies of the same bytes; skipping them avoids ~3x/6x bloat.
SONAME_RE = re.compile(r"^lib(opencv_\w+|ade)\.so\.\d+$")

DIST_NAME = "opencv-contrib-python"
DIST_NAME_NORM = "opencv_contrib_python"


def urlsafe_b64_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def record_line(path: str, data: bytes | None) -> str:
    if data is None:
        return f"{path},,\n"
    digest = urlsafe_b64_nopad(hashlib.sha256(data).digest())
    return f"{path},sha256={digest},{len(data)}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-packages-dir", required=True, help="Dir containing the cmake-installed cv2/ package")
    parser.add_argument("--install-prefix", required=True, help="OpenCV CMAKE_INSTALL_PREFIX (its lib/ holds libopencv_*.so)")
    parser.add_argument("--version", required=True, help="Wheel version, e.g. 5.0.0")
    parser.add_argument("--python-tag", required=True, help="e.g. cp312")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cv2_src = Path(args.python_packages_dir) / "cv2"
    if not cv2_src.is_dir():
        raise SystemExit(f"cv2 package not found at {cv2_src} — did the cmake install step run?")

    shared_libs_by_name: dict[str, Path] = {}
    for lib_dir in (Path(args.install_prefix) / "lib", Path(args.install_prefix) / "lib64"):
        if not lib_dir.is_dir():
            continue
        for f in lib_dir.iterdir():
            if SONAME_RE.match(f.name) and f.name not in shared_libs_by_name:
                shared_libs_by_name[f.name] = f
    shared_libs = sorted(shared_libs_by_name.values())
    if not shared_libs:
        raise SystemExit(f"no libopencv_*.so.<N> found under {args.install_prefix}/lib{{,64}}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = out_dir / f"{DIST_NAME_NORM}-{args.version}-{args.python_tag}-{args.python_tag}-linux_x86_64.whl"

    dist_info = f"{DIST_NAME_NORM}-{args.version}.dist-info"
    records: list[str] = []

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # cv2/ — the built python package, as cmake produced it (loader shims included).
        # config.py is regenerated below (portable path) instead of copied verbatim.
        for f in sorted(cv2_src.rglob("*")):
            if f.is_dir() or "__pycache__" in f.parts or f.name == "config.py":
                continue
            arcname = f"cv2/{f.relative_to(cv2_src)}"
            data = f.read_bytes()
            zf.writestr(arcname, data)
            records.append(record_line(arcname, data))

        # cv2/config.py — replace cmake's absolute-path version with one relative
        # to the installed package, so the wheel is portable across machines/users.
        config_py = (
            "import os\n\n"
            "BINARIES_PATHS = [\n"
            "    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cv2.libs')\n"
            "] + BINARIES_PATHS\n"
        ).encode()
        zf.writestr("cv2/config.py", config_py)
        records.append(record_line("cv2/config.py", config_py))

        # cv2.libs/ — bundle the shared libs cv2.so links against, next to the package
        # ponytail: no rpath/auditwheel patching here; relies on cv2's own BINARIES_PATHS
        # loader plus `make run-cuda`'s LD_LIBRARY_PATH as a fallback. Upgrade to full
        # auditwheel repair if a bare `uv run` (no LD_LIBRARY_PATH) needs to work too.
        for so in shared_libs:
            arcname = f"cv2.libs/{so.name}"
            data = so.read_bytes()
            zf.writestr(arcname, data)
            records.append(record_line(arcname, data))

        metadata = (
            "Metadata-Version: 2.1\n"
            f"Name: {DIST_NAME}\n"
            f"Version: {args.version}\n"
            "Summary: OpenCV (contrib) built from source with CUDA support\n"
        ).encode()
        zf.writestr(f"{dist_info}/METADATA", metadata)
        records.append(record_line(f"{dist_info}/METADATA", metadata))

        wheel_meta = (
            "Wheel-Version: 1.0\n"
            "Generator: build-opencv-cuda.sh\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {args.python_tag}-{args.python_tag}-linux_x86_64\n"
        ).encode()
        zf.writestr(f"{dist_info}/WHEEL", wheel_meta)
        records.append(record_line(f"{dist_info}/WHEEL", wheel_meta))

        top_level = b"cv2\n"
        zf.writestr(f"{dist_info}/top_level.txt", top_level)
        records.append(record_line(f"{dist_info}/top_level.txt", top_level))

        records.append(record_line(f"{dist_info}/RECORD", None))
        zf.writestr(f"{dist_info}/RECORD", "".join(records))

    print(f"wrote {wheel_path}")


if __name__ == "__main__":
    main()
