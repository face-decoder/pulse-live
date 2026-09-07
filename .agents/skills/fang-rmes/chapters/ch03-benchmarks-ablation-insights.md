# Chapter 3: Benchmarks, Latency Profiling & Ablation Insights

## Core Idea
Empirical evaluations on CAS(ME)2 and SAMM Long Videos prove RMES achieves state-of-the-art F1 performance (0.1489 and 0.1667) with the lowest computational complexity (0.6M FLOPs, 161k parameters), while ablation experiments establish the critical necessity of face alignment, 3rd-level pyramid scaling, and 10Hz FIR lowpass temporal filtering.

---

## Frameworks Introduced

- **Long-Video Micro-Expression Spotting Evaluation Protocol**:
  - *IoU Thresholding*: A detected temporal interval $W_{\\text{pred}} = [P_n - K, P_n + K]$ is declared a True Positive (TP) if:
    $$\\text{IoU}(W_{\\text{pred}}, W_{\\text{gt}}) = \\frac{|W_{\\text{pred}} \\cap W_{\\text{gt}}|}{|W_{\\text{pred}} \\cup W_{\\text{gt}}|} \\ge 0.5$$
  - *Dataset Precision & Recall Aggregation*:
    $$R = \\frac{\\sum_{i=1}^V a_i}{\\sum_{i=1}^V X_i}, \\quad P = \\frac{\\sum_{i=1}^V a_i}{\\sum_{i=1}^V Y_i}, \\quad F_1 = \\frac{2PR}{P+R}$$
    where $V$ is number of videos, $X_i$ is GT micro-expressions count, $Y_i$ is spotted intervals count, and $a_i$ is TP count.

---

## Key Concepts

- **CAS(ME)2 Dataset**: 98 spontaneous long videos from 22 subjects recorded at 30 FPS ($640 \\times 480$ resolution), containing 57 annotated micro-expressions (average duration ~200ms, $K=6$).
- **SAMM Long Videos Dataset**: 147 high-speed long videos from 32 subjects recorded at 200 FPS ($2040 \\times 1088$ resolution), containing 159 micro-expressions (average duration ~235ms, $K=47$).
- **FLOPs Efficiency Advantage**: RMES requires 0.6M FLOPs, compared to 2.1M (Liong et al. 2021), 1.4M (Liong et al. 2022), 2.7M (Liong et al. 2023), and 810M (Yu et al. 2021 LSSNet).

---

## Benchmark Comparisons

### Table II: SOTA Micro-Expression Spotting Benchmark (F1 Scores)

| Method | Front-End Representation | Back-End Network | CAS(ME)2 ($F_1$) | SAMM Long Videos ($F_1$) |
| :--- | :--- | :--- | :--- | :--- |
| **Duque et al. (2018)** | Riesz Phase Variance (Averaged) | Region Peak Thresholding | 0.0806 | 0.0711 |
| **LSSNet (Yu et al., 2021)** | Optical Flow | 2-Stream Deep CNN (23M params)| 0.0420 | 0.1310 |
| **MESNet (Wang et al., 2021)** | Optical Flow | Multi-scale CNN | 0.0360 | 0.0880 |
| **Yang et al. (2021)** | Appearance + Facial Action Units | AU Temporal Fusion | 0.0153 | 0.1155 |
| **Yap et al. (2022)** | Appearance | 3D-CNN (End-to-End) | 0.0714 | 0.0466 |
| **Liong et al. (2021)** | Optical Flow | 3-Stream Shallow CNN | 0.1173 | 0.1520 |
| **LSSNet (Liong et al., 2022)**| Optical Flow | Multi-temporal Stream Net | 0.0808 | 0.0878 |
| **Spot-then-Recognize (2023)**| Optical Flow | Optical Flow + Shallow CNN | 0.1214 | 0.0949 |
| **RMES (Fang et al., 2023)** | **Riesz Quaternionic Phase** | **3-Stream Shallow CNN** | **0.1489 (+26.9%)** | **0.1667 (+9.7%)** |

---

## Model Complexity & Latency

### Table IV: Model Complexity Comparison

| Architecture | # Parameters | # FLOPs | Inference Latency | Preprocessing Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Yu et al. (2021)** | 23,000,000 | 810.0 M | ~0.085 s | ~0.120 s |
| **Liong et al. (2021)** | 315,000 | 2.1 M | 0.0020 s | 0.120 s (TV-L1) |
| **Liong et al. (2023)** | 161,000 | 2.7 M | 0.0022 s | 0.120 s (TV-L1) |
| **RMES (Ours)** | **161,000** | **0.6 M** | **0.0019 s** | **0.0266 s (Riesz)** |

---

## Detailed Ablation Studies

### 1. The Critical Role of Face Alignment (FA)
- *Finding*: Without Face Alignment, global head rotations, speech posture shifts, and breathing create sweeping phase differences across the face.
- *Quantitative Impact*:
  - **CAS(ME)2**: FA increases Precision from 0.0814 $\\to$ 0.1069, improving $F_1$ from 0.1223 $\\to$ 0.1489 (+21.7% gain).
  - **SAMM Long Videos**: False positives drop from 341 $\\to$ 171 (a 50% reduction in FP), boosting Precision from 0.0833 $\\to$ 0.1493 and $F_1$ from 0.1168 $\\to$ 0.1667 (+42.7% gain).

### 2. Riesz Pyramid Level Selection
- **Level 1 & 2 (Fine Scales)**: High noise sensitivity and rapid phase wrap-around ($> \\pi$ radians) cause score degradation.
- **Level 3 (Mid Scale)**: **Optimal**. Maximizes $F_1$ score by balancing displacement range with spatial resolution.
- **Level 4 (Coarse Scale)**: Resolution drops to $13 \\times 13$, obliterating spatial localization and dropping $F_1$.

### 3. Temporal Filter Strategy
- **FIR Lowpass ($f_c = 10\\text{ Hz}$)**: Yields highest $F_1$ across datasets.
- **Bandpass ($2\\text{--}10\\text{ Hz}$)**: Significantly degrades performance because eliminating $<2\\text{ Hz}$ removes the slow-moving baseline and introduces Gibbs ringing near interval boundaries.

### 4. FACS ROI Cropping vs Full Face
- Restricting input to Eyebrows + Mouth ($30 \\times 30$) consistently outperforms Full Face ($42 \\times 42$ or $224 \\times 224$) because jawline translation, neck movement, and hair boundaries are excluded.

---

## Key Takeaways
1. Phase features yield a +26.94% $F_1$ boost over optical flow on the exact same shallow CNN backbone structure.
2. Front-end preprocessing is the true computational bottleneck of video emotion pipelines; Riesz transform cuts preprocessing latency from 120ms to 26.6ms.
3. Strict landmark-based face alignment is mandatory when utilizing phase representations to prevent head-pose false triggers.

---

## Connects To
- **Ch 1: Riesz Pyramid & Phase Representation**: Explains the mathematical mechanism behind the phase advantage.
- **Ch 2: Pipeline Architecture**: Details the implementation of each ablated module.
