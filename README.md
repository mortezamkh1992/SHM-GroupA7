# Semi-Supervised and Unsupervised Learning for Health Indicator Extraction from Guided Waves in Aerospace Composite Structures

**Diversity-DeepSAD vs. Degradation-Trend-Constrained VAE**

This repository contains the **PyTorch and TensorFlow implementation** accompanying the paper:

> *Semi-supervised and unsupervised learning for health indicator extraction from guided waves in aerospace composite structures*

The work proposes a **data-driven framework for learning robust health indicators (HIs)** from guided-wave measurements in composite aerospace structures.

---

## Abstract

Health indicators (HIs) are central to diagnosing and prognosing the condition of aerospace composite structures, enabling efficient maintenance and operational safety. However, extracting reliable HIs remains challenging due to variability in material properties, stochastic damage evolution, and diverse damage modes. Manufacturing defects (e.g. disbonds) and in-service incidents (e.g. bird strikes) further complicate this process.

This study presents a comprehensive data-driven framework that learns HIs via two learning approaches integrated with multi-domain signal processing. Because ground-truth HIs are unavailable, a semi-supervised and an unsupervised approach are proposed:  
(i) a **Diversity-DeepSAD** approach augmented with continuous auxiliary labels used as hypothetical damage proxies, enabling modelling of intermediate degradation states; and  
(ii) a **Degradation-Trend-Constrained Variational Autoencoder (DTC-VAE)**, in which monotonicity is enforced via an explicit trend constraint.

Guided waves with multiple excitation frequencies are used to monitor single-stiffener composite structures under fatigue loading. Time, frequency, and time–frequency representations are explored, and per-frequency HIs are fused via unsupervised ensemble learning to mitigate frequency dependence and reduce variance. Using fast Fourier transform features, the models achieved fitness scores of **81.6 % (Diversity-DeepSAD)** and **92.3 % (DTC-VAE)**, indicating improved monotonicity and consistency over existing baselines.  The proposed history-independent framework, supported by prognostic-metrics-guided Bayesian optimisation and frequency-agnostic HI fusion, enables the estimation of more robust HIs for aeronautical composite structures.

---

## Overview of the Proposed Framework

<p align="center">
  <img src="figures/SHM-diagram.png" width="100%">
</p>

*Overview of the signal processing, learning, and health-indicator fusion pipeline.*

---

## Repository Contents

- **Entry points / orchestration**
  - `main.py` - main experiment entry point (runs signal processing, training and evaluation pipeline).
  - `loop.py` - loop utility for hyperparameter optimisation and sensitivity analyses.

- **Learning models**
  - `DeepSAD.py` - Diversity-DeepSAD model implementation and training/inference logic.
  - `VAE.py` - DTC-VAE model implementation and training/inference logic.
  - `WAE.py` - Weight Average Ensemble model used for frequency fusion.

- **Signal processing & feature extraction**
  - `Extract_features.py` - feature extraction utilities (mean, variance, skewness, kurtosis...).
  - `Signal_Processing/` - signal processing utilities (FFT, STFT, HLB, EMD).

- **Metrics & plotting**
  - `Prognostic_criteria.py` - prognostic/HI quality metrics (monotonicity, trendability, prognosability and fitness scoring).
  - `Graphs.py` - plotting/visualization utilities for results.

---

## Datasets

The experimental data consists of guided-wave measurements acquired from PZT sensors on composite structures under fatigue loading.

**Download:**  
https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/QNURER

**Dataset preparation:**
1. Download all PZT datasets.
2. Extract the ZIP files.
3. Rename each dataset folder to one of:
- PZT-L1-03
- PZT-L1-04
- PZT-L1-05
- PZT-L1-09
- PZT-L1-23
4. Place all PZT folders in the **same directory**, containing no other files.

---

## Citation

If you use this code or dataset in your work, please cite the paper:

```bibtex
@article{AUTHOR_YEAR_HI_GW,
title   = {Semi-supervised and unsupervised learning for health indicator extraction from guided waves in aerospace composite structures},
author  = {AUTHOR LIST},
journal = {Journal Name},
year    = {YEAR},
}
```

A preprint is available on **arXiv**:  
https://arxiv.org/abs/2510.24614

---