import cv2
import numpy as np

from src.api.video_process import _extract_jpeg_frames
from src.image.modules import decode_jpeg


def _jpeg_bytes(color=(0, 0, 255)) -> bytes:
    img = np.full((4, 4, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_decode_jpeg_roundtrip():
    payload = _jpeg_bytes()
    img = decode_jpeg(payload)
    assert img is not None
    assert img.shape == (4, 4, 3)


def test_decode_jpeg_invalid_returns_none():
    assert decode_jpeg(b"not an image") is None


def test_extract_single_frame():
    jpg = _jpeg_bytes()
    frames, rest = _extract_jpeg_frames(b"junk" + jpg + b"tail")
    assert len(frames) == 1
    assert frames[0] == jpg
    assert rest == b"tail"


def test_extract_multiple_frames():
    a, b = _jpeg_bytes((255, 0, 0)), _jpeg_bytes((0, 255, 0))
    frames, rest = _extract_jpeg_frames(a + b)
    assert frames == [a, b]
    assert rest == b""


def test_extract_incomplete_frame_is_buffered():
    jpg = _jpeg_bytes()
    partial = jpg[: len(jpg) // 2]
    frames, rest = _extract_jpeg_frames(partial)
    if frames:
        assert rest == b""
    else:
        assert rest == partial
