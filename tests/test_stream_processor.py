import numpy as np
import pytest

from src.api.roi_grid import RoiGrid
from src.api.window_buffers import WindowBuffers
from src.api.window_snapshot import WindowSnapshot


def test_window_snapshot_n_frames():
    snapshot = WindowSnapshot(
        mags=[0.1],
        flows=[],
        bboxes=[None, {"x": 1}],
        received_at=0.0,
        webrtc_ms=[],
        landmark_ms=[],
        flow_ms=[],
        timestamps=[1.0, 2.0],
    )
    assert snapshot.n_frames == 2


def test_roi_grid_bounds_layout():
    grid = RoiGrid(n_roi=5)
    h, w = RoiGrid.TILE
    assert grid.bounds(0) == (0, h, 0, w)
    assert grid.bounds(2) == (0, h, 2 * w, 3 * w)
    assert grid.bounds(3) == (h, 2 * h, 0, w)


def test_roi_grid_pack_unpack_roundtrip():
    grid = RoiGrid(n_roi=5)
    t = 3
    h, w = RoiGrid.TILE
    expected = np.random.rand(t, grid.n_roi, 2, h, w).astype(np.float32)

    flows = [expected[0][idx].transpose(1, 2, 0) for idx in range(grid.n_roi)]
    mag, canvas = grid.pack(flows)
    assert canvas.shape == (grid.rows * h, RoiGrid.COLS * w, 2)

    expected_mag = float(
        np.mean([np.mean(np.hypot(f[..., 0], f[..., 1])) for f in flows])
    )
    assert mag == pytest.approx(expected_mag)

    result = grid.unpack([canvas] * t)
    assert result.shape == (t, grid.n_roi, 2, h, w)
    np.testing.assert_allclose(result[0], expected[0])


def test_roi_grid_unpack_empty_raises():
    try:
        RoiGrid(n_roi=5).unpack([])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty flow list")


def test_window_buffers_ready_and_snapshot():
    buffers = WindowBuffers()
    assert not buffers.ready

    for i in range(20):
        buffers.record_flow(mag=float(i), canvas=np.zeros((4, 4, 2)), flow_ms=1.0)
        buffers.record_frame(None, webrtc_ms=2.0, landmark_ms=3.0, at=float(i))

    assert buffers.ready
    snapshot = buffers.snapshot(received_at=99.0)
    assert len(snapshot.mags) == 20
    assert snapshot.received_at == 99.0
    assert all(ts in snapshot.timestamps for ts in range(20))
