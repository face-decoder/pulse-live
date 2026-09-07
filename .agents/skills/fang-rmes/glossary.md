# RMES Glossary of Terms & Mathematical Symbols

Alphabetical index of terminology and notation from Fang et al. (2023).

---

- **$A_m$ (Local Amplitude)**: Magnitude of the monogenic quaternion signal $r_m = \\sqrt{I_m^2 + R_{1m}^2 + R_{2m}^2}$, measuring local contrast energy.
- **Apex**: The instant of maximum muscle displacement during a facial expression.
- **Aperture Problem**: The ambiguity in measuring motion orthogonal to 1D texture gradients; optical flow solves this via spatial smoothing, whereas Riesz phase tracks orientation explicitly.
- **CAS(ME)2**: Spontaneous Chinese Academy of Sciences Macro- and Micro-Expression dataset recorded at 30 FPS.
- **Eulerian Motion Magnification (EMM)**: A technique that amplifies subtle color and motion variations in video using temporal bandpass filtering.
- **FACS (Facial Action Coding System)**: Anatomical taxonomy of facial muscle movements decomposed into Action Units (AUs).
- **$F_1$ Score**: Harmonic mean of Precision and Recall: $F_1 = 2PR / (P+R)$, evaluated with $\\text{IoU} \\ge 0.5$.
- **$H_x, H_y$ (Riesz Quadrature Filters)**: Spatial-domain 2D Hilbert transform pair with frequency responses $-i\\frac{\\omega_x}{\\|\\omega\\|}$ and $-i\\frac{\\omega_y}{\\|\\omega\\|}$.
- **$K$ (Accumulation Interval)**: Half the average duration of micro-expressions in frames; used to accumulate inter-frame phase differences.
- **Laplacian Pyramid**: Multi-scale representation where each level stores bandpass difference images between successive Gaussian pyramid stages.
- **LOSO (Leave-One-Subject-Out)**: Cross-validation where one participant is held out for testing per fold.
- **Macro-Expression**: Voluntary facial expression lasting 0.5s to 4.0s with large spatial amplitude.
- **Micro-Expression (ME)**: Involuntary, subtle facial movement lasting 1/25s to 1/2s (40ms to 500ms).
- **Monogenic Signal**: 2D quaternion extension of 1D analytic signal: $(I, R_1, R_2)$.
- **Non-Causal FIR Filter**: Finite Impulse Response filter operating with zero group delay across past and future frames in batch mode.
- **Onset**: The initial moment facial muscles begin contraction from neutral.
- **Offset**: The moment facial muscles return to baseline neutral posture.
- **Optical Flow**: Vector field $(u, v)$ estimating apparent pixel displacement across frames.
- **$\\phi_m$ (Local Phase)**: Angular component of the monogenic signal representing local structural position.
- **Quaternionic Phase Difference ($\\Delta\\Phi\\cos\\Theta, \\Delta\\Phi\\sin\\Theta$)**: Orientation-invariant vector measuring inter-frame displacement in radians.
- **Riesz Pyramid**: Multi-scale pyramid computing Riesz transforms at each Laplacian band.
- **RMES**: Real-Time Micro-Expression Spotting framework (Fang et al., 2023).
- **SAMM Long Videos**: Spontaneous Actions and Micro-Movements high-speed dataset recorded at 200 FPS.
- **$\\theta_m$ (Local Orientation)**: Angle of dominant image gradient measured with respect to the horizontal axis.
