import math

import numpy as np


class RoiGrid:
    TILE: tuple[int, int] = (64, 64)
    MARGIN: float = 0.05
    COLS: int = 3

    def __init__(self, n_roi: int) -> None:
        self.n_roi = n_roi
        self.rows = math.ceil(n_roi / self.COLS)

    def bounds(self, idx: int) -> tuple[int, int, int, int]:
        h, w = self.TILE
        row, col = divmod(idx, self.COLS)
        return row * h, (row + 1) * h, col * w, (col + 1) * w

    @property
    def blank(self) -> np.ndarray:
        return np.zeros((self.TILE[1], self.TILE[0], 3), dtype=np.uint8)

    def pack(self, flows: list[np.ndarray]) -> tuple[float, np.ndarray]:
        h, w = self.TILE
        canvas = np.zeros((self.rows * h, self.COLS * w, 2), dtype=np.float32)
        mags = []
        for idx, flow in enumerate(flows):
            y1, y2, x1, x2 = self.bounds(idx)
            canvas[y1:y2, x1:x2, :] = flow
            mags.append(float(np.mean(np.hypot(flow[..., 0], flow[..., 1]))))
        return float(np.mean(mags)), canvas

    def unpack(self, canvases: list[np.ndarray]) -> np.ndarray:
        frames = []
        for raw in canvases:
            canvas = np.asarray(raw, dtype=np.float32)
            tiles = []
            for idx in range(self.n_roi):
                y1, y2, x1, x2 = self.bounds(idx)
                tiles.append(canvas[y1:y2, x1:x2, :].transpose(2, 0, 1))
            frames.append(np.stack(tiles, axis=0))

        if not frames:
            raise ValueError("No flow canvases to assemble")

        return np.stack(frames, axis=0)
