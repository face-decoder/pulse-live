# Chapter 2: RMES Pipeline Architecture & Training

## Core Idea
The RMES pipeline unifies landmark-based face alignment, zero-phase-shift FIR temporal lowpass filtering, K-interval quaternionic phase accumulation, FACS-based ROI feature cropping, and a 3-stream asymmetric CNN to spot micro-expression intervals in real time ($0.0285\\text{s}$ per frame).

---

## Frameworks Introduced

- **3-Stage RMES Pipeline**:
  1. **Preprocessing**:
     - *Face Alignment*: OpenFace 2.0 68-landmarks detect facial bounding box, linear warping normalizes tilt/scale to $224 \\times 224$.
     - *Riesz Pyramid Subband*: 3rd level Laplacian subband processed via Riesz filter pair.
     - *Temporal Lowpass FIR*: Non-causal FIR filter ($f_c = 10\\text{ Hz}$, $100\\text{ms}$ time constant) with zero group delay filters noise without Gibbs ringing.
     - *K-Frame Phase Accumulation*: Inter-frame phase differences accumulated over $K = \\text{round}(\\text{avg\\_duration}/2)$ frames ($K=6$ for 30fps CAS(ME)2, $K=47$ for 200fps SAMMLV):
       $$\\Delta\\Phi_m \\cos\\Theta_m = \\sum_{k=0}^{K-1} \\Delta\\phi_{m-k}\\cos\\theta_{m-k}, \\quad \\Delta\\Phi_m \\sin\\Theta_m = \\sum_{k=0}^{K-1} \\Delta\\phi_{m-k}\\sin\\theta_{m-k}$$
       $$|\\Delta\\Phi_m| = \\sqrt{(\\Delta\\Phi_m \\cos\\Theta_m)^2 + (\\Delta\\Phi_m \\sin\\Theta_m)^2}$$
     - *FACS ROI Cropping & Stacking*: Eyebrow region ($15 \\times 30$) and mouth region ($15 \\times 30$) cropped according to FACS Action Units, stacked into a $30 \\times 30 \\times 3$ tensor, and normalized via Z-score.
  2. **3-Stream Asymmetric CNN**:
     - Three parallel Conv2D streams with unequal filter counts to match facial muscle kinematics:
       - Stream 1 ($\\Delta\\Phi\\cos\\Theta$, horizontal): 3 filters ($3 \\times 3$, stride 1) + MaxPool ($6 \\times 6$, stride 6) $\\to 5 \\times 5 \\times 3$
       - Stream 2 ($\\Delta\\Phi\\sin\\Theta$, vertical): 5 filters ($3 \\times 3$, stride 1) + MaxPool ($6 \\times 6$, stride 6) $\\to 5 \\times 5 \\times 5$ (higher allocation for vertical muscle twitches)
       - Stream 3 ($|\\Delta\\Phi|$, magnitude): 8 filters ($3 \\times 3$, stride 1) + MaxPool ($6 \\times 6$, stride 6) $\\to 5 \\times 5 \\times 8$
     - *Feature Fusion*: Concatenates streams to $5 \\times 5 \\times 16 = 400$-d vector $\\to$ FC1 ($400 \\to 400$, ReLU) $\\to$ FC2 ($400 \\to 1$) $\\to$ raw frame likelihood score $s_i$.
  3. **Postprocessing & Peak Detection**:
     - *Moving Average Smoothing*: Over window length $2K+1$:
       $$\\hat{s}_i = \\frac{1}{2K+1} \\sum_{j=i-K}^{i+K} s_j$$
     - *Dynamic Peak Detection*: Finds peaks $P_n$ exceeding adaptive threshold $H$:
       $$H = \\hat{s}_{\\text{mean}} + h \\times (\\hat{s}_{\\text{max}} - \\hat{s}_{\\text{mean}}), \\quad h = 0.7$$
       with minimum horizontal distance between adjacent peaks $\\ge k$.
     - *Spotted Interval Output*: $[P_n - K, P_n + K]$.

---

## Key Concepts

- **FACS-Guided ROI**: Facial Action Coding System identifies eyebrows (AU1, AU2, AU4) and mouth (AU12, AU14, AU15, AU20) as the primary sites of micro-expression leakage; excluding cheeks and jawline eliminates non-expressive head artifacts.
- **LOSO (Leave-One-Subject-Out) Cross-Validation**: Validation protocol where all video clips of one subject are withheld for testing while the remaining subjects train the CNN, ensuring subject-independent generalization.
- **Intersection Over Union (IoU) Ground Truth Assignment**: For training sample at frame $i$, target $S_i = 1$ if $\\text{IoU}([i-K, i+K], [T_{\\text{onset}}, T_{\\text{offset}}]) \\ge 0.5$, else $0$.
- **MSE Loss on Continuous Scores**: Trained with Mean Squared Error $L = \\frac{1}{N} \\sum_{i=1}^N (s_i - S_i)^2$ to predict a continuous likelihood bell curve peaking at the apex.

---

## Mental Models

- **"K is the Onset-to-Apex Bridge"**: Micro-expressions rise from neutral to apex in roughly half their total duration. Accumulating phase over $K$ frames captures the maximum net displacement without requiring full trajectory tracking.
- **"Asymmetric Filters Match Facial Anatomy"**: Human facial expressions (e.g. brow lowering, lip corner pulling, chin raising) generate predominantly vertical image displacement; giving 5 filters to vertical phase vs 3 to horizontal maximizes representational capacity.

---

## Anti-patterns

- **Temporal Bandpass Filtering (EMM Style)**: Bandpass filters ($2\\text{--}10\\text{ Hz}$) cut off low frequencies ($<2\\text{ Hz}$) that represent the slow baseline drift and onset ramp, causing ringing oscillations that trigger false peaks.
- **Symmetric CNN Filter Allocation**: Giving identical filter budgets to horizontal and vertical channels wastes parameters on horizontal channels that carry less discriminative signal.
- **Static Peak Thresholds**: Using a fixed absolute threshold (e.g. $s > 0.5$) fails across subjects due to individual baseline muscle tone variations; adaptive dynamic range thresholding ($H = \\hat{s}_{\\text{mean}} + h(\\hat{s}_{\\text{max}} - \\hat{s}_{\\text{mean}})$) is essential.

---

## Code Examples

### PyTorch RMES 3-Stream Shallow CNN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMES3StreamCNN(nn.Module):
    """
    RMES 3-Stream Shallow CNN for Micro-Expression Spotting (Fang et al., 2023).
    Inputs: Tensor of shape (B, 3, 30, 30) where channels are:
      - Channel 0: Horizontal Phase Difference (dPhi * cos(Theta))
      - Channel 1: Vertical Phase Difference (dPhi * sin(Theta))
      - Channel 2: Magnitude of Phase Difference (|dPhi|)
    Output: Scalar frame likelihood score s in [0, 1]
    """
    def __init__(self):
        super().__init__()
        # Stream 1: Horizontal motion (3 filters)
        self.conv_h = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        
        # Stream 2: Vertical motion (5 filters - higher capacity for facial anatomy)
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
        
        # Stream 3: Magnitude (8 filters)
        self.conv_m = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        # 6x6 max pool with stride 6 -> downsamples 30x30 to 5x5
        self.pool = nn.MaxPool2d(kernel_size=6, stride=6)
        
        # Total concatenated features: (3 + 5 + 8) * 5 * 5 = 16 * 25 = 400
        self.fc1 = nn.Linear(400, 400)
        self.fc2 = nn.Linear(400, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split channels
        x_h = x[:, 0:1, :, :]
        x_v = x[:, 1:2, :, :]
        x_m = x[:, 2:3, :, :]
        
        # Convolutions + MaxPool + ReLU
        f_h = self.pool(F.relu(self.conv_h(x_h)))  # (B, 3, 5, 5)
        f_v = self.pool(F.relu(self.conv_v(x_v)))  # (B, 5, 5, 5)
        f_m = self.pool(F.relu(self.conv_m(x_m)))  # (B, 8, 5, 5)
        
        # Concatenate across channel dimension
        f_cat = torch.cat([f_h, f_v, f_m], dim=1)  # (B, 16, 5, 5)
        f_flat = f_cat.view(f_cat.size(0), -1)      # (B, 400)
        
        out = F.relu(self.fc1(f_flat))
        score = self.fc2(out)                      # (B, 1)
        return score
```
- **What it demonstrates**: The exact PyTorch implementation of the 161k-parameter RMES CNN backbone.

---

## Reference Tables

### Pipeline Stage Latency & Complexity Breakdown

| Pipeline Stage | Algorithm / Layer | Latency (sec) | % of Total Time | FLOPs |
| :--- | :--- | :--- | :--- | :--- |
| **Face Alignment** | OpenFace 2.0 (68 landmarks + affine warp) | 0.0180s | 63.2% | ~10M |
| **Riesz Transform & Phase**| 2D Quadrature Filters (Spatial Subband 3) | 0.0086s | 30.2% | ~2.5M |
| **Temporal FIR & Accum**| Non-causal Lowpass ($f_c=10\\text{Hz}$) + $K$-sum | < 0.0005s | ~1.5% | < 0.1M |
| **Shallow CNN Inference** | 3-Stream Conv2D + FC Layers | 0.0019s | 6.7% | **0.6M** |
| **Postprocessing** | Moving Average + Adaptive Peak Detect | < 0.0001s | < 0.5% | Negligible |
| **Overall System** | End-to-End Real-Time Spotting | **0.0285s (35.1 FPS)**| **100.0%** | **~13.2M total** |

---

## Worked Example: End-to-End Frame Interval Spotting Walkthrough

1. **Input Video**: 30 FPS clip from CAS(ME)2 with $T=300$ frames.
2. **Preprocessing**:
   - Landmark tracking detects face, warps to $224 \\times 224$.
   - Laplacian subband 3 extracted; Riesz phase differences calculated.
   - FIR lowpass filtered ($f_c = 10\\text{ Hz}$); accumulated over $K=6$ frames ($200\\text{ms}$).
   - ROI cropper extracts eyebrow ($15 \\times 30$) and mouth ($15 \\times 30$) $\\to 30 \\times 30 \\times 3$.
3. **Inference**:
   - Forward pass yields score sequence $s = [s_1, s_2, \\dots, s_{294}]$.
4. **Postprocessing**:
   - Moving average window ($2K+1 = 13$ frames) produces smoothed scores $\\hat{s}$.
   - Video statistics: $\\hat{s}_{\\text{mean}} = 0.082$, $\\hat{s}_{\\text{max}} = 0.840$.
   - Adaptive threshold: $H = 0.082 + 0.7 \\times (0.840 - 0.082) = 0.6126$.
   - Peak detected at frame $P = 142$ with score $0.791 > H$.
   - Spotted interval produced: $[P-K, P+K] = [142-6, 142+6] = [136, 148]$.
   - Ground truth interval was $[134, 150]$.
   - $\\text{IoU} = \\frac{|[136, 148] \\cap [134, 150]|}{|[136, 148] \\cup [134, 150]|} = \\frac{12}{16} = 0.75 \\ge 0.5$ $\\implies$ **True Positive!**

---

## Key Takeaways
1. Preprocessing accounts for 93.3% of the total time (with Face Alignment being 68% of preprocessing); optimizing front-end feature representation yields much larger speedups than pruning back-end neural networks.
2. The 3-stream shallow CNN requires only 161k parameters and 0.6M FLOPs, running inference in 1.9ms.
3. Adaptive peak thresholding ($h=0.7$) eliminates the need for per-subject manual calibration.

---

## Connects To
- **Ch 1: Riesz Pyramid & Phase Representation**: Details the mathematical derivation of the 3 input channels.
- **Ch 3: Benchmarks & Ablation Insights**: Reviews quantitative results on CAS(ME)2 and SAMM Long Videos.
