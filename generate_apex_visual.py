import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# Generate synthetic magnitude signal representing micro-expression flow
np.random.seed(42)
x = np.arange(0, 100)
# Create a peak at frame 50
peak_pos = 50
signal_base = 0.05 + 0.02 * np.random.randn(100)
# Gaussian bump for the micro-expression peak
signal_peak = 0.8 * np.exp(-((x - peak_pos) / 10) ** 2)
signal = signal_base + signal_peak
# Ensure positive signal
signal = np.clip(signal, 0.01, None)

# Smooth as in ApexWindowDetector
smoothed = gaussian_filter1d(signal, sigma=1.5)

# Calculate threshold (e.g. 70th percentile)
threshold = np.percentile(smoothed, 70)

# Find onset using similar logic
# From peak (50) backwards, find where it drops below 30% of peak value or starts rising
peak_val = smoothed[peak_pos]
ratio = 0.30
onset = peak_pos
while onset > 1:
    if smoothed[onset] < peak_val * ratio:
        break
    if smoothed[onset] < smoothed[onset - 1]:
        # Local minimum or rising
        break
    onset -= 1

# Include context window (e.g., 5 frames) as in code: left = max(0, left - context)
context = 5
onset_with_context = max(0, onset - context)

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

# Plot signals
ax.plot(x, signal, label='Raw Magnitude Signal', color='#cbd5e1', linewidth=1.5, linestyle='--')
ax.plot(x, smoothed, label='Smoothed Signal ($\sigma=1.5$)', color='#1e3a8a', linewidth=2.5)

# Plot threshold line
ax.axhline(y=threshold, color='#dc2626', linestyle=':', label='Detection Threshold', linewidth=1.8)

# Highlight peaks and boundaries
ax.scatter([peak_pos], [smoothed[peak_pos]], color='#1e3a8a', s=100, zorder=5)
ax.annotate('Apex Frame\n(Peak)', xy=(peak_pos, smoothed[peak_pos]), xytext=(peak_pos + 4, smoothed[peak_pos] + 0.1),
            arrowprops=dict(facecolor='#1e3a8a', shrink=0.08, width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color='#1a365d')

ax.scatter([onset], [smoothed[onset]], color='#059669', s=80, zorder=5)
ax.annotate('Onset Frame\n(Start)', xy=(onset, smoothed[onset]), xytext=(onset - 18, smoothed[onset] + 0.15),
            arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
            fontsize=10, fontweight='bold', color='#065f46')

# Context offset annotation
ax.axvline(x=onset_with_context, color='#0284c7', linestyle='--', linewidth=1.2)
ax.annotate('Onset with Context\n(Frame 37)', xy=(onset_with_context, 0.2), xytext=(onset_with_context - 25, 0.35),
            arrowprops=dict(arrowstyle="->", color='#0284c7', connectionstyle="arc3,rad=-0.2"),
            fontsize=9, color='#0369a1')

# Shaded area representing the selected window
ax.fill_between(x[onset_with_context:peak_pos+1], smoothed[onset_with_context:peak_pos+1], 
                color='#3b82f6', alpha=0.2, label='Selected Onset-to-Apex Jendela')

# Title and labels
ax.set_title('Mekanisme Deteksi Jendela Temporal (Onset-to-Apex)', fontsize=14, fontweight='bold', pad=15, color='#1e293b')
ax.set_xlabel('Index Bingkai (Frames)', fontsize=11, labelpad=8, color='#334155')
ax.set_ylabel('Magnitude / Aliran Optik Terfilter', fontsize=11, labelpad=8, color='#334155')

# Customize grid and limits
ax.set_xlim(10, 80)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)

plt.tight_layout()
plt.savefig('onset_to_apex_visualization.png', dpi=300)
print("Visualization generated successfully.")
