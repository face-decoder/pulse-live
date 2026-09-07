# Article Documentation Directory (`article/`)

This directory contains research articles, benchmark evaluation documentation, Jupyter notebooks, and visual figures comparing facial micro-expression spotting and valence classification across international standard datasets (**CAS(ME)²**, **CASME II**, and **SAMM**) using both baseline **SVM** and **Deep Learning (1D CNN & CNN-Transformer)** architectures.

## Directory Structure

```
article/
├── README.md                      # This file
├── docs/                          # Documentation & reports
│   ├── benchmark.md              # Multi-dataset benchmark report
│   └── result.md                 # Research results guide
├── notebooks/                     # Jupyter notebooks organized by dataset
│   ├── casme2/                   # CAS(ME)² notebooks
│   ├── casme_ii/                 # CASME II notebooks
│   ├── samm/                     # SAMM notebooks
│   ├── spotting_apex_visualization.ipynb
│   └── spotting_iou_visualization.ipynb
├── scripts/                       # Standalone experiment scripts
│   ├── casme2/                   # CAS(ME)² scripts
│   ├── casme_ii/                 # CASME II scripts
│   └── samm/                     # SAMM scripts
├── src/                           # Article-specific utility modules
│   ├── __init__.py
│   ├── label.py                  # MELabel: emotion label mapper
│   ├── sequence.py               # FrameSequence: frame clip manager
│   ├── spotting.py               # MESpotting: apex and boundary spotting
│   └── profiler.py               # RealTimeProfiler: latency and FPS profiler
├── data/                          # Experiment data files
│   ├── me_inference_casme_ii.csv
│   └── me_inference_samm.csv
└── assets/                        # Images and visual assets
    ├── iou_visual.png
    └── docs/
        └── casme2_iou_visual.png
```

## Research Notebooks

- 📓 **CAS(ME)²** (`notebooks/casme2/`):
  - Baseline SVM: [`me_spotting_cas_me_2.ipynb`](notebooks/casme2/me_spotting_cas_me_2.ipynb)
  - Deep Learning (CNN & Transformer): [`me_spotting_cas_me_2_dl.ipynb`](notebooks/casme2/me_spotting_cas_me_2_dl.ipynb)
- 📓 **CASME II** (`notebooks/casme_ii/`):
  - Baseline SVM: [`me_spotting_casme_ii.ipynb`](notebooks/casme_ii/me_spotting_casme_ii.ipynb)
  - Deep Learning (CNN & Transformer): [`me_spotting_casme_ii_dl.ipynb`](notebooks/casme_ii/me_spotting_casme_ii_dl.ipynb)
- 📓 **SAMM** (`notebooks/samm/`):
  - Baseline SVM: [`me_spotting_samm.ipynb`](notebooks/samm/me_spotting_samm.ipynb)
  - Deep Learning (CNN & Transformer): [`me_spotting_samm_dl.ipynb`](notebooks/samm/me_spotting_samm_dl.ipynb)

## Documents & Assets

- 📄 **[benchmark.md](docs/benchmark.md)**: Multi-dataset benchmark report covering **CAS(ME)²**, **CASME II**, and **SAMM** (Spotting F1, LOSO Accuracy/F1, Latency).
- 🖼️ **[iou_visual.png](assets/iou_visual.png)**: Visual plot illustrating Ground Truth vs Spotted temporal windows and IoU computation ($\ge 0.5$ TP standard).

## Quick Summary Metrics Table

| Dataset | Frame Rate | Micro Samples | Spotting F1 ($\text{IoU} \ge 0.5$) | Best Model (LOSO Acc) | SVM Macro F1 | 1D CNN Macro F1 | CNN-Transformer Macro F1 | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CAS(ME)²** | **30 FPS** | **57** | **0.8421** | **1D CNN (0.6809)** | 0.4823 | 0.4051 | 0.4024 | **114,200 FPS** |
| **CASME II** | **200 FPS** | **239** | **0.3013** | **SVM (0.6639)** | **0.4936** | 0.4842 | 0.4728 | **40,600 FPS** |
| **SAMM** | **200 FPS** | **136** | **0.7206** | **SVM (0.6691)** | **0.6120** | 0.4937 | 0.4712 | **117,700 FPS** |
