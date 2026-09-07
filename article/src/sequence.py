from __future__ import annotations

import os
from typing import Sequence, Any
import numpy as np
import torch


class FrameSequence:
    """Manages micro-expression frame clips, optical flow tensors, and ROI magnitudes."""

    def __init__(
        self,
        subject: str,
        clip_name: str,
        flow: np.ndarray | torch.Tensor,
        magnitudes: Sequence[float] | np.ndarray | None = None,
        label: str | None = None,
        fps: int = 200,
    ) -> None:
        self.subject = subject
        self.clip_name = clip_name
        self.fps = fps
        self.label = label

        if isinstance(flow, torch.Tensor):
            self.flow = flow.detach().cpu().numpy()
        else:
            self.flow = np.asarray(flow, dtype=np.float32)

        if magnitudes is not None:
            self.magnitudes = [float(m) for m in magnitudes]
        else:
            self.magnitudes = self._calc_magnitudes()

    def _calc_magnitudes(self) -> list[float]:
        """Compute average flow magnitude per frame."""
        if len(self.flow) == 0:
            return []
        # Expecting flow shape (T, N_roi, 2, H, W) or (T, H, W, 2)
        if self.flow.ndim == 5:
            # (T, N_roi, 2, H, W)
            mags_per_roi = np.hypot(self.flow[:, :, 0, :, :], self.flow[:, :, 1, :, :])
            return [float(np.mean(mags_per_roi[t])) for t in range(len(self.flow))]
        elif self.flow.ndim == 4:
            # (T, H, W, 2)
            mags = np.hypot(self.flow[..., 0], self.flow[..., 1])
            return [float(np.mean(mags[t])) for t in range(len(self.flow))]
        return [0.0] * len(self.flow)

    def clip(self, onset: int, offset: int) -> FrameSequence:
        """Slice sequence temporally from onset to offset (inclusive).

        Args:
            onset: Start frame index.
            offset: End frame index (inclusive).

        Returns:
            New sliced FrameSequence instance.
        """
        if len(self.flow) == 0:
            return FrameSequence(
                subject=self.subject,
                clip_name=self.clip_name,
                flow=np.empty_like(self.flow),
                magnitudes=[],
                label=self.label,
                fps=self.fps,
            )

        start = max(0, int(onset))
        end = min(len(self.flow) - 1, int(offset))

        if start > end:
            sliced_flow = np.empty((0, *self.flow.shape[1:]), dtype=self.flow.dtype)
            sliced_mags = []
        else:
            sliced_flow = self.flow[start : end + 1]
            sliced_mags = self.magnitudes[start : end + 1] if self.magnitudes else []

        return FrameSequence(
            subject=self.subject,
            clip_name=self.clip_name,
            flow=sliced_flow,
            magnitudes=sliced_mags,
            label=self.label,
            fps=self.fps,
        )

    def to_tensor(
        self, device: str | torch.device = "cpu", max_len: int | None = None
    ) -> torch.Tensor:
        """Convert flow array to 5D PyTorch tensor (1, N_roi*C, T, H, W) for 3D CNN.

        Returns:
            torch.Tensor with shape (1, 10, T, H, W).
        """
        flow_arr = self.flow
        if max_len is not None and len(flow_arr) > max_len:
            flow_arr = flow_arr[:max_len]

        if flow_arr.ndim == 5:
            # (T, N_roi, 2, H, W) -> (N_roi, 2, T, H, W) -> (1, N_roi*2, T, H, W)
            T, N_roi, C, H, W = flow_arr.shape
            tensor = (
                torch.from_numpy(flow_arr.astype(np.float32))
                .permute(1, 2, 0, 3, 4)
                .reshape(N_roi * C, T, H, W)
                .unsqueeze(0)
            )
        elif flow_arr.ndim == 4:
            # (T, H, W, 2) -> (2, T, H, W) -> (1, 2, T, H, W)
            T, H, W, C = flow_arr.shape
            tensor = (
                torch.from_numpy(flow_arr.astype(np.float32))
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
            )
        else:
            tensor = torch.from_numpy(flow_arr.astype(np.float32)).unsqueeze(0)

        return tensor.to(device)

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        subject: str = "",
        clip_name: str = "",
        label: str | None = None,
        fps: int = 200,
    ) -> FrameSequence | None:
        """Load FrameSequence from precomputed .npz file."""
        if not os.path.exists(npz_path):
            return None
        data = np.load(npz_path)
        flow = data["flow"]
        magnitudes = data["magnitudes"].tolist() if "magnitudes" in data else None
        return cls(
            subject=subject,
            clip_name=clip_name,
            flow=flow,
            magnitudes=magnitudes,
            label=label,
            fps=fps,
        )

    @staticmethod
    def pad_batch(
        sequences: Sequence[FrameSequence],
        max_len: int | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Batch and pad multiple FrameSequences into (B, N_roi*C, max_t, H, W)."""
        valid_seqs = [s for s in sequences if len(s.flow) > 0]
        if not valid_seqs:
            return torch.empty(0, device=device)

        lengths = [len(s.flow) for s in valid_seqs]
        target_len = max_len or max(lengths)
        
        tensors = []
        for s in valid_seqs:
            t = s.to_tensor(device=device) # (1, channels, T, H, W)
            if t.shape[2] < target_len:
                pad_t = target_len - t.shape[2]
                t = torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad_t))
            elif t.shape[2] > target_len:
                t = t[:, :, :target_len, :, :]
            tensors.append(t)

        return torch.cat(tensors, dim=0)

    def __len__(self) -> int:
        return len(self.flow)
