# Chapter 1: Riesz Pyramid & Phase-Based Motion Representation

## Core Idea
Micro-expressions (MEs) are subtle, non-rigid, localized facial motions (lasting 1/25 to 1/2 second) that standard optical flow smooths away due to spatial regularization; spatial-domain Riesz Pyramids extract quaternionic phase differences that preserve local motion variations with 77.8% lower computational cost.

---

## Frameworks Introduced

- **Spatial-Domain Quaternionic Phase Difference**:
  - *Formulation*: Constructs a Laplacian pyramid, computes a 2D steerable Hilbert (Riesz) transformer in quadrature along spatial dimensions, formulates the triple as a monogenic quaternion $r_m = I_m + i R_{1m} + j R_{2m}$, and measures motion between adjacent frames as the quaternionic log difference $\\log(\\hat{r}_m \\hat{r}_{m-1}^{-1}) \\approx i \\Delta \\phi_m \\cos \\theta_m + j \\Delta \\phi_m \\sin \\theta_m$.
  - *When to use*: Detecting subtle, rapid, non-rigid localized deformations (e.g., facial action units, micro-expressions, pulse-induced skin vibrations) where global smoothing or optical flow blur boundaries.
  - *How*:
    1. Build spatial Laplacian pyramid subbands $I_m$.
    2. Filter with quadrature Riesz pair ($H_x, H_y$).
    3. Construct unit quaternion $\\hat{r}_m$.
    4. Compute inter-frame angular deviation via quaternion conjugate multiplication.

- **Phase-vs-Optical-Flow Aperture Tradeoff**:
  - *Optical Flow*: Imposes spatial smoothness constraints across wide pixel neighborhoods to solve the aperture problem; effective for rigid translation, but destroys subtle high-frequency non-rigid deformations.
  - *Riesz Phase*: Maintains independent, per-pixel local phase estimates $\\phi_m$ and orientations $\\theta_m$; noisier, but retains raw high-frequency non-rigid micro-movements.

---

## Key Concepts

- **Micro-Expression (ME)**: Involuntary facial muscle contraction lasting 1/25s to 1/2s (40ms–500ms) with small spatial amplitude, consisting of three phases: onset, apex, and offset.
- **Monogenic Signal**: A 2D generalization of the 1D analytic signal, combining image intensity $I$ and its 2D Riesz transform responses $(R_1, R_2)$ into a 3-tuple $(I, R_1, R_2)$.
- **Quaternionic Representation ($r_m$)**: $r_m = A_m \\cos \\phi_m + i A_m \\sin \\phi_m \\cos \\theta_m + j A_m \\sin \\phi_m \\sin \\theta_m$, parameterized by local amplitude $A_m$, local orientation $\\theta_m$, and local phase $\\phi_m$.
- **Quaternionic Phase Invariance**: The pair $(\\phi_m \\cos \\theta_m, \\phi_m \\sin \\theta_m)$ is invariant to the sign ambiguity between $(\\phi_m, \\theta_m)$ and $(-\\phi_m, \\theta_m + \\pi)$.
- **Pyramid Level Scale Trade-off**: Lower levels (fine scale) have higher spatial resolution and sensitivity to minute shifts, but limited displacement range and higher risk of phase wrap-around ($>\\pi$); higher levels (coarser scale) handle larger displacements but lose spatial detail.

---

## Mental Models

- **"Think of Phase as Local Micro-Displacement Probes"**: Rather than tracking feature vectors or computing pixel correspondences across regions, phase measures the local shift of image structures directly in radians along the dominant gradient orientation.
- **"Smoothness is the Enemy of Micro-Signals"**: Global spatial regularizers in optical flow treat micro-expressions as noise and smooth them out. Local phase preserves the true muscle twitch.

---

## Anti-patterns

- **Spatial Averaging of Phase over Broad Regions (Duque et al.)**: Spatially averaging phase across entire eye/mouth bounding boxes destroys local intra-region gradient differences (e.g. inner vs outer corner movement).
- **Double-Extraction Pipeline (EMM + Optical Flow)**: Running Eulerian Motion Magnification (EMM) followed by optical flow extracts motion twice, wasting compute and introducing Gibbs ringing artifacts from temporal bandpass filtering.
- **Frequency-Domain Steerable Pyramids (MiMaMo)**: Computing complex steerable pyramids in the frequency domain is ~4× slower and prone to phase wrap-around artifacts compared to spatial Riesz pyramids.

---

## Code Examples

### Quaternionic Phase Difference from Image Pair (NumPy / PyTorch)

```python
import numpy as np

def riesz_transform_2d(image: np.ndarray):
    """
    Compute 2D spatial Riesz transform using approx quadrature filters.
    Transfer functions in frequency domain: -i * (omega_x / |omega|) and -i * (omega_y / |omega|)
    """
    h, w = image.shape
    u = np.fft.fftfreq(h)[:, None]
    v = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1.0  # avoid division by zero

    H_x = -1j * (u / radius)
    H_y = -1j * (v / radius)
    H_x[0, 0] = 0.0
    H_y[0, 0] = 0.0

    F = np.fft.fft2(image)
    R1 = np.real(np.fft.ifft2(F * H_x))
    R2 = np.real(np.fft.ifft2(F * H_y))
    return R1, R2

def compute_quaternion_phase_diff(I_prev: np.ndarray, I_curr: np.ndarray):
    """
    Calculates quaternionic phase difference components: (dPhi_x, dPhi_y, |dPhi|)
    """
    R1_prev, R2_prev = riesz_transform_2d(I_prev)
    R1_curr, R2_curr = riesz_transform_2d(I_curr)

    # Amplitudes
    A_prev = np.sqrt(I_prev**2 + R1_prev**2 + R2_prev**2) + 1e-8
    A_curr = np.sqrt(I_curr**2 + R1_curr**2 + R2_curr**2) + 1e-8

    # Unit quaternions (q0, q1, q2) = (I, R1, R2) / A
    q0_p, q1_p, q2_p = I_prev / A_prev, R1_prev / A_prev, R2_prev / A_prev
    q0_c, q1_c, q2_c = I_curr / A_curr, R1_curr / A_curr, R2_curr / A_curr

    # Conjugate product: r_curr * conjugate(r_prev)
    d_x = q1_c * q0_p - q0_c * q1_p  # i component ~ dPhi * cos(theta)
    d_y = q2_c * q0_p - q0_c * q2_p  # j component ~ dPhi * sin(theta)
    d_mag = np.sqrt(d_x**2 + d_y**2)

    return d_x, d_y, d_mag
```
- **What it demonstrates**: Exact computation of the 3-channel quaternionic motion representation ($\\Delta\\Phi\\cos\\Theta, \\Delta\\Phi\\sin\\Theta, |\\Delta\\Phi|$) from consecutive frames.

---

## Reference Tables

### Motion Representation Comparison for Micro-Expression Spotting

| Property | Optical Flow (TV-L1, Farneback) | Spatially Averaged Phase (Duque) | RMES Quaternionic Phase |
| :--- | :--- | :--- | :--- |
| **Domain** | Spatial displacement $(u, v)$ | Scalar phase variance per region | 3-Channel Dense Phase $(\\Delta\\Phi_x, \\Delta\\Phi_y, \\|\\Delta\\Phi\\|)$ |
| **Spatial Resolution** | Smoothed / Blurred across pixels | Coarse (3 scalar values per face) | Dense per-pixel resolution |
| **Non-rigid Motion Capture**| Poor (suppressed by smoothness term)| Poor (averaged over entire ROI) | **High** (preserves local twitch vectors) |
| **Computational Speed** | Slow (~120ms / frame) | Fast (~15ms / frame) | **Fast (~26.6ms / frame, incl. alignment)** |
| **F1 Score on CAS(ME)2** | 0.1173 (Liong 2021) | 0.0806 (Duque 2018) | **0.1489 (+26.9% improvement)** |

---

## Worked Example: Phase vs Optical Flow in Asymmetric Mouth Elevation

Consider an asymmetric micro-expression where a subject involuntarily elevates only the right corner of the mouth:
1. **Optical Flow Analysis**: The global regularization term attempts to minimize the gradient of the vector field $\\int (||\\nabla u||^2 + ||\\nabla v||^2) dx dy$. As a result, the right lip vector propagates into the cheek and left lip, causing a uniform upward motion vector field that blurs the onset.
2. **Quaternionic Riesz Phase Analysis**:
   - The right oral commissure pixels yield high vertical phase shifts ($\\Delta \\Phi_y = \\Delta \\Phi \\sin \\Theta > 0.8\\text{ rad}$).
   - The central lip and left oral commissure maintain near-zero phase shifts ($|\\Delta \\Phi| < 0.05\\text{ rad}$).
   - The 3-channel map preserves the exact gradient boundary, providing the downstream CNN with high-contrast localized activation.

---

## Key Takeaways
1. Optical flow is optimized for rigid movement and large displacements; it degrades micro-expression detection by over-smoothing non-rigid muscle actions.
2. Riesz transform computes 2D steerable Hilbert pairs in the spatial domain, avoiding complex frequency-domain FFT overhead.
3. Unit quaternionic phase differences provide an orientation-invariant, wrap-resistant representation of inter-frame motion.
4. Level 3 of the Laplacian pyramid strikes the optimal empirical balance between spatial frequency sensitivity and displacement range.

---

## Connects To
- **Ch 2: RMES Pipeline Architecture**: Explains how these 3-channel phase maps are filtered, accumulated over $K$ frames, and fed into the 3-stream CNN.
- **Ch 3: Benchmarks & Ablation Insights**: Quantifies the empirical gains of phase over optical flow and isolates the role of pyramid scales.
