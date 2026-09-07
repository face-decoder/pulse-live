from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Sequence, Any, Generator
import numpy as np
import pandas as pd


class RealTimeProfiler:
    """Measures latency breakdown, sequence-level runtime, and real-time FPS throughput."""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def record(self, stage: str) -> Generator[None, None, None]:
        """Context manager to measure runtime of a stage in milliseconds."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.records[stage].append(elapsed_ms)

    def add(self, stage: str, duration_ms: float) -> None:
        """Manually record a latency observation in milliseconds."""
        self.records[stage].append(float(duration_ms))

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self.records.clear()

    def stats(
        self, frame_counts: Sequence[int] | None = None
    ) -> dict[str, Any]:
        """Compute summary statistics, per-frame latency, and FPS throughput."""
        stage_means = {
            stage: float(np.mean(times)) if times else 0.0
            for stage, times in self.records.items()
        }
        total_latency = sum(stage_means.values())

        if frame_counts and len(frame_counts) > 0:
            avg_frames = float(np.mean(frame_counts))
        else:
            avg_frames = 1.0

        avg_frame_latency = (
            total_latency / avg_frames if avg_frames > 0 else total_latency
        )
        fps_throughput = (
            1000.0 / avg_frame_latency if avg_frame_latency > 0 else 0.0
        )

        return {
            "stages": stage_means,
            "total_seq_latency_ms": total_latency,
            "avg_seq_length_frames": avg_frames,
            "avg_frame_latency_ms": avg_frame_latency,
            "fps_throughput": fps_throughput,
        }

    def report(
        self, frame_counts: Sequence[int] | None = None
    ) -> pd.DataFrame:
        """Generate formatted latency and throughput DataFrame."""
        summary = self.stats(frame_counts)
        rows = []
        for stage, avg_ms in summary["stages"].items():
            rows.append(
                {
                    "Metric": f"Latency: {stage.capitalize()}",
                    "Value": f"{avg_ms:.3f} ms / seq",
                }
            )

        rows.extend(
            [
                {
                    "Metric": "Total Sequence Latency",
                    "Value": f"{summary['total_seq_latency_ms']:.3f} ms",
                },
                {
                    "Metric": "Average Sequence Length",
                    "Value": f"{summary['avg_seq_length_frames']:.1f} frames",
                },
                {
                    "Metric": "Estimated Frame Latency",
                    "Value": f"{summary['avg_frame_latency_ms']:.3f} ms / frame",
                },
                {
                    "Metric": "Throughput (FPS)",
                    "Value": f"{summary['fps_throughput']:.1f} FPS",
                },
            ]
        )
        return pd.DataFrame(rows)
