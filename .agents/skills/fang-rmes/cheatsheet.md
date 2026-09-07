# RMES Quick Reference Cheatsheet & Decision Rules

A decision-making reference and debugging guide for implementing and tuning RMES.

---

## 1. Decision Rules: When to Use What

- **Use Spatial Riesz Phase When**:
  - The target motion is subtle, localized, non-rigid, and brief (< 500ms).
  - Low latency / real-time operation is required (> 30 FPS on embedded/edge hardware).
  - Spatial gradient boundaries must remain sharp without region-wide bleeding.
- **Use Optical Flow Instead When**:
  - Objects are rigid and displacements exceed several pixels per frame.
  - Full-frame dense flow vectors are required across featureless textures.
- **Use FIR Lowpass Filtering ($f_c = 10\\text{Hz}$) When**:
  - The motion is transient/non-periodic (like muscle twitches).
  - Avoiding Gibbs ringing near onset/offset is critical.
- **Set $K = \\text{round}(\\text{FPS} \\times 0.2 / 2)$**:
  - Standard ME duration is ~200ms. Half-duration corresponds to 100ms.
  - At 30 FPS $\\to K = 6$. At 200 FPS $\\to K = 47$.

---

## 2. Key Hyperparameters Reference

| Symbol | Parameter | Value (CAS(ME)2) | Value (SAMM LV) | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **$L$** | Laplacian Pyramid Level | Level 3 | Level 3 | Maximizes displacement range before spatial resolution degrades |
| **$f_c$** | Lowpass Cutoff Frequency | 10 Hz | 10 Hz | Eliminates high-frequency camera noise; $100\\text{ms}$ response time |
| **$K$** | Accumulation Half-Window | 6 frames | 47 frames | Exactly spans onset-to-apex duration (~100ms) |
| **$h$** | Peak Threshold Factor | 0.7 | 0.7 | Adaptive threshold: $H = \\mu + h(\\text{max} - \\mu)$ |
| **$k$** | Min Peak Distance | $K$ (6) | $K$ (47) | Prevents multiple detections of the same micro-expression |
| **$W_{\\text{in}}$** | CNN Input Dimension | $30 \\times 30 \\times 3$ | $30 \\times 30 \\times 3$ | Stacked Eyebrows ($15\\times 30$) and Mouth ($15\\times 30$) |
| **$F_{\\text{alloc}}$**| CNN Stream Filter Budget| 3 / 5 / 8 | 3 / 5 / 8 | 3 horizontal, 5 vertical (anatomical priority), 8 magnitude |

---

## 3. Tensor Dimension Checklist

```
Raw Frame (RGB/Grayscale) -> Face Alignment & Linear Warp -> (224, 224)
Laplacian Level 3 Subband -> (56, 56)
Riesz Quadrature Filtering -> Monogenic Quaternions -> Inter-frame Phase Diffs (2, 56, 56)
Temporal FIR Filter -> K-Frame Accumulation -> (3, 56, 56) [dPhi_x, dPhi_y, |dPhi|]
FACS ROI Crop (Eyebrows + Mouth) & Resample -> (3, 30, 30)
Stream 1 (dPhi_x): Conv2D(1->3) + MaxPool(6x6) -> (3, 5, 5)
Stream 2 (dPhi_y): Conv2D(1->5) + MaxPool(6x6) -> (5, 5, 5)
Stream 3 (|dPhi|): Conv2D(1->8) + MaxPool(6x6) -> (8, 5, 5)
Concatenate & Flatten -> Vector of length 400
FC1(400 -> 400) + FC2(400 -> 1) -> Scalar score s_i in [0, 1]
```

---

## 4. Troubleshooting & Diagnostic Matrix

| Symptom | Probable Cause | Fix |
| :--- | :--- | :--- |
| **Excessive False Positives** | Missing or inaccurate face alignment | Use 68-point landmark tracker (e.g. OpenFace) with affine stabilization |
| **Phase Wrap-Around Noise** | Using Pyramid Level 1 or 2 with large motion | Switch to Level 3 or increase scale downsampling |
| **Missed Low-Intensity Twitches** | Using Pyramid Level 4 (coarse scale) | Downsample less (Level 3 is optimal) |
| **Ringing Around Apex Frames** | Using Bandpass filter ($2-10\\text{Hz}$) | Replace with zero-phase FIR lowpass filter ($f_c=10\\text{Hz}$) |
| **Subject-Specific Score Bias** | Using fixed score threshold ($s > 0.5$) | Use dynamic range formula $H = \\hat{s}_{\\text{mean}} + 0.7(\\hat{s}_{\\text{max}} - \\hat{s}_{\\text{mean}})$ |
