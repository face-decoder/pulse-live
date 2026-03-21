import numpy as np


class FeatureExtractor:
    """Extract optical-flow features from ROI frame sequences.

    Attributes:
        ROI_ORDER: Canonical ordering of the five facial ROIs.
        BILATERAL_PAIRS: Left–right ROI pairs for asymmetry features.
        TOTAL_DIM: Total feature vector dimensionality (78).
    """

    ROI_ORDER: list[str] = [
        "left_eye", "right_eye", "lips", "left_eyebrow", "right_eyebrow",
    ]
    BILATERAL_PAIRS: list[tuple[str, str]] = [
        ("left_eye", "right_eye"),
        ("left_eyebrow", "right_eyebrow"),
    ]

    K_APEX = 3
    PER_ROI = 4
    N_ROI = 5

    SCALAR_DIM = N_ROI * PER_ROI * K_APEX          # 60
    ASYM_DIM = len(BILATERAL_PAIRS) * 3 * K_APEX    # 18
    TOTAL_DIM = SCALAR_DIM + ASYM_DIM                # 78

    def __init__(self, k_apex: int = K_APEX) -> None:
        """Initialise the extractor.

        Args:
            k_apex: Number of top apex frames to use.
        """
        self.k_apex = k_apex

    # ── public: extract ───────────────────────────────────────────────

    def extract(
        self,
        roi_frames: list[list[dict]],
        apex_indices: np.ndarray,
        phases: dict,
    ) -> np.ndarray:
        """Extract a feature vector from a single clip.

        Args:
            roi_frames: ``loaded["frames"]`` from the ``.npy`` file.
                ``roi_frames[t]`` is a list of dicts, each with keys
                ``"roi"``, ``"dx"``, ``"dy"``.
            apex_indices: Output of
                ``ApexPhase.find_top_k_apex()``.
            phases: Output of ``ApexPhase.find_phase()``.

        Returns:
            Feature vector of shape ``(78,)`` as float32.
        """
        if len(apex_indices) == 0:
            return np.zeros(self.TOTAL_DIM, dtype=np.float32)

        total_frames = len(roi_frames)
        scalar_blocks: list[np.ndarray] = []
        asym_blocks: list[np.ndarray] = []

        for apex_idx in apex_indices[: self.k_apex]:
            phase = phases.get(int(apex_idx), {})
            onset, apex_end = self._safe_bounds(phase, total_frames)
            roi_feats = self._extract_apex(
                roi_frames, onset, apex_end, total_frames,
            )

            scalar_blocks.append(
                np.concatenate([roi_feats[r] for r in self.ROI_ORDER]),
            )
            asym_blocks.append(self._bilateral(roi_feats))

        # Pad to k_apex if fewer apex frames were found
        while len(scalar_blocks) < self.k_apex:
            scalar_blocks.append(
                np.zeros(self.N_ROI * self.PER_ROI, dtype=np.float32),
            )
            asym_blocks.append(np.zeros(6, dtype=np.float32))

        return np.concatenate(scalar_blocks + asym_blocks).astype(np.float32)

    # ── public: augment ───────────────────────────────────────────────

    def augment(
        self,
        feat: np.ndarray,
        scale_p: float = 0.5,
        noise_p: float = 0.3,
        flip_p: float = 0.5,
    ) -> np.ndarray:
        """Augment a feature vector.  Call ONLY on training data.

        Three independent augmentations, each applied with its own
        probability:

        - **scale** — multiply magnitudes by a random factor in
          ``[0.7, 1.3]``.
        - **noise** — add small Gaussian noise (σ = 0.02).
        - **flip**  — simulate a horizontal flip of the face.

        Args:
            feat: Feature vector of shape ``(78,)``.
            scale_p: Probability of applying scale augmentation.
            noise_p: Probability of adding Gaussian noise.
            flip_p: Probability of applying horizontal flip.

        Returns:
            Augmented feature vector (same shape, float32).
        """
        feat = feat.copy()
        if np.random.random() < scale_p:
            feat = self._augment_scale(feat)
        if np.random.random() < noise_p:
            feat += np.random.normal(0, 0.02, feat.shape).astype(np.float32)
        if np.random.random() < flip_p:
            feat = self._augment_flip(feat)
        return feat

    # ── private: extraction ───────────────────────────────────────────

    def _extract_apex(
        self,
        roi_frames: list[list[dict]],
        onset: int,
        apex_end: int,
        total_frames: int,
    ) -> dict[str, np.ndarray]:
        """Compute per-ROI scalar features for one apex window."""
        dx_buf: dict[str, list[np.ndarray]] = {r: [] for r in self.ROI_ORDER}
        dy_buf: dict[str, list[np.ndarray]] = {r: [] for r in self.ROI_ORDER}

        for t in range(onset, min(apex_end + 1, total_frames)):
            frame_map = {item["roi"]: item for item in roi_frames[t]}
            for roi in self.ROI_ORDER:
                item = frame_map.get(roi)
                if item is not None:
                    dx_buf[roi].append(
                        np.asarray(item["dx"], dtype=np.float32),
                    )
                    dy_buf[roi].append(
                        np.asarray(item["dy"], dtype=np.float32),
                    )

        roi_feats: dict[str, np.ndarray] = {}
        for roi in self.ROI_ORDER:
            if dx_buf[roi]:
                dx_seg = np.stack(dx_buf[roi], axis=0)
                dy_seg = np.stack(dy_buf[roi], axis=0)
                roi_feats[roi] = self._roi_features(dx_seg, dy_seg)
            else:
                roi_feats[roi] = np.zeros(self.PER_ROI, dtype=np.float32)
        return roi_feats

    def _roi_features(
        self, dx_seg: np.ndarray, dy_seg: np.ndarray,
    ) -> np.ndarray:
        """Compute 4 scalar features from dx/dy segments for one ROI."""
        if dx_seg.shape[0] == 0:
            return np.zeros(self.PER_ROI, dtype=np.float32)
        mag = np.sqrt(dx_seg ** 2 + dy_seg ** 2)
        return np.array([
            mag.mean(),
            mag[-1].max(),
            dx_seg.mean(),
            dy_seg.mean(),
        ], dtype=np.float32)

    def _bilateral(self, roi_feats: dict[str, np.ndarray]) -> np.ndarray:
        """Compute bilateral asymmetry features from left–right pairs."""
        parts: list[float] = []
        for left, right in self.BILATERAL_PAIRS:
            fl = roi_feats.get(left, np.zeros(self.PER_ROI, dtype=np.float32))
            fr = roi_feats.get(right, np.zeros(self.PER_ROI, dtype=np.float32))
            parts.extend([fl[0] - fr[0], fl[2] - fr[2], fl[3] - fr[3]])
        return np.array(parts, dtype=np.float32)

    # ── private: augmentation ─────────────────────────────────────────

    def _augment_scale(self, feat: np.ndarray) -> np.ndarray:
        """Scale magnitude features by a random factor."""
        scale = np.random.uniform(0.7, 1.3)
        for k in range(self.k_apex):
            for r in range(self.N_ROI):
                base = k * self.N_ROI * self.PER_ROI + r * self.PER_ROI
                feat[base] *= scale
                feat[base + 1] *= scale
        return feat

    def _augment_flip(self, feat: np.ndarray) -> np.ndarray:
        """Simulate a horizontal flip by negating dx and swapping pairs."""
        # Negate net_dx for every ROI
        for k in range(self.k_apex):
            for r in range(self.N_ROI):
                feat[k * self.N_ROI * self.PER_ROI + r * self.PER_ROI + 2] *= -1

        # Swap left/right ROI features
        roi_idx = {r: i for i, r in enumerate(self.ROI_ORDER)}
        swaps = [
            (roi_idx["left_eye"], roi_idx["right_eye"]),
            (roi_idx["left_eyebrow"], roi_idx["right_eyebrow"]),
        ]
        for k in range(self.k_apex):
            bk = k * self.N_ROI * self.PER_ROI
            for left, right in swaps:
                sl = slice(
                    bk + left * self.PER_ROI,
                    bk + left * self.PER_ROI + self.PER_ROI,
                )
                sr = slice(
                    bk + right * self.PER_ROI,
                    bk + right * self.PER_ROI + self.PER_ROI,
                )
                feat[sl], feat[sr] = feat[sr].copy(), feat[sl].copy()

        # Negate delta_dx in asymmetry block
        n_pairs = len(self.BILATERAL_PAIRS)
        asym_base = self.SCALAR_DIM
        for k in range(self.k_apex):
            for p in range(n_pairs):
                feat[asym_base + k * n_pairs * 3 + p * 3 + 1] *= -1
        return feat

    # ── private: utility ──────────────────────────────────────────────

    @staticmethod
    def _safe_bounds(phase: dict, total_frames: int) -> tuple[int, int]:
        """Clamp onset/apex indices to valid frame range.

        Args:
            phase: Phase dict with optional ``onset``/``start`` and
                ``apex`` keys.
            total_frames: Total number of frames in the clip.

        Returns:
            A ``(onset, apex_end)`` tuple of clamped indices.
        """
        onset = phase.get("onset", phase.get("start", 0))
        apex = phase.get("apex", onset + 1)
        onset = int(np.clip(onset, 0, max(total_frames - 1, 0)))
        apex = int(np.clip(apex, 0, max(total_frames - 1, 0)))
        if apex <= onset:
            apex = min(total_frames, onset + 1)
        return onset, apex
