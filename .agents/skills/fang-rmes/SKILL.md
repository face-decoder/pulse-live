---
name: fang-rmes
description: "Real-time micro-expression spotting (RMES) using spatial-domain Riesz pyramid quaternionic phase differences and 3-stream shallow CNNs (Fang et al., 2023). Covers motion phase representation, FIR temporal filtering, FACS ROI cropping, LOSO cross-validation, and real-time spotting workflows."
---

# RMES: Real-Time Micro-Expression Spotting

A specialized skill for implementing, analyzing, and applying the **RMES (Real-Time Micro-Expression Spotting)** framework developed by Fang et al. (HKUST / Ydentity / Bright Nation, 2023).

---

## Quick Navigation

- **[Chapter 1: Riesz Pyramid & Phase-Based Motion Representation](chapters/ch01-riesz-pyramid-phase-representation.md)**
  *Monogenic signal, spatial Riesz transform, quaternionic phase differences, aperture problem & why phase outperforms optical flow.*
- **[Chapter 2: RMES Pipeline Architecture & Training](chapters/ch02-rmes-pipeline-architecture.md)**
  *OpenFace 2.0 alignment, FIR temporal lowpass filtering, K-interval phase accumulation, FACS ROI cropping, 3-stream CNN, and peak detection.*
- **[Chapter 3: Benchmarks, Latency Profiling & Ablation Insights](chapters/ch03-benchmarks-ablation-insights.md)**
  *CAS(ME)2 & SAMM Long Videos results, 0.6M FLOPs latency breakdown, face alignment ablation, pyramid level trade-offs, and temporal filter comparisons.*
- **[Design Patterns & Algorithms](patterns.md)**
  *Reusable implementations: Quaternionic Phase Extractor, K-Step Accumulator, 3-Stream Asymmetric CNN, and Adaptive Threshold Peak Detector.*
- **[Quick Cheatsheet & Decision Rules](cheatsheet.md)**
  *Hyperparameter selection matrix, tensor dimensions, decision trees, and diagnostic troubleshooting checklist.*
- **[Comprehensive Glossary](glossary.md)**
  *Complete index of mathematical terms, acronyms, and operational definitions.*

---

## Core Tenets of RMES

1. **Phase Over Optical Flow for Micro-Motions**: Spatial regularization in optical flow destroys subtle, non-rigid localized muscle twitches. Local Riesz phase preserves raw displacement gradients with 77.8% lower computational cost.
2. **Face Alignment is Non-Negotiable**: Phase is hyper-sensitive. Without landmark-based rigid alignment (OpenFace 68 landmarks), head tilt and translation induce massive false-positive phase shifts.
3. **Temporal Lowpass beats Bandpass**: Unlike Eulerian Motion Magnification (which uses bandpass filters and suffers from Gibbs ringing), micro-expressions are non-periodic. A zero-group-delay FIR lowpass filter ($f_c = 10\text{ Hz}$) removes sensor noise while preserving onset/offset transitions.
4. **Asymmetric Filter Allocation**: Facial anatomy exhibits predominantly vertical deformations during expressions. Allocating more CNN filters to vertical phase ($\Delta\Phi\sin\Theta$) than horizontal ($\Delta\Phi\cos\Theta$) optimizes parameter efficiency.
5. **K-Step Interval Accumulation**: Choosing $K = \text{round}(\text{avg\_duration} / 2)$ bridges the onset-to-apex temporal span while remaining within the $(-\pi, \pi]$ phase unwrapping interval.

---

## Recommended Default Parameters

| Parameter | CAS(ME)2 (30 FPS) | SAMM Long Videos (200 FPS) | Description |
| :--- | :--- | :--- | :--- |
| **Pyramid Level** | Level 3 | Level 3 | 3rd Laplacian subband (optimal scale/noise balance) |
| **Filter Cutoff ($f_c$)** | 10 Hz | 10 Hz | Non-causal FIR lowpass filter |
| **Accumulation ($K$)** | 6 frames (~200ms ME) | 47 frames (~235ms ME) | Half average ME duration |
| **ROI Crop Size** | 15×30 (Eyebrows & Mouth) | 15×30 (Eyebrows & Mouth) | Stacked to 30×30 feature map |
| **Peak Threshold ($h$)** | 0.7 | 0.7 | Adaptive threshold coefficient |
| **Min Peak Dist ($k$)** | $K$ (6) | $K$ (47) | Minimum frame distance between spots |
