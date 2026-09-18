# Lab 02: Impact of Spatial Image Filtering on Skin-Lesion Classification

## Project Overview
This project investigates how five different spatial-domain image filters impact the classification performance of three leading Convolutional Neural Network (CNN) backbones (ResNet50, EfficientNet-B0, and DenseNet121). The evaluation is conducted on a 5-class subset of the ISIC 2019 dataset, comprising 24,211 dermoscopic images. 

## Experimental Setup
* **Models Evaluated:** Top 3 performing architectures from Lab 1: ResNet50 (81.19%), EfficientNet-B0 (81.08%), and DenseNet121 (80.95%).
* **Dataset:** ISIC 2019 (5-class subset: MEL, NV, BCC, BKL, AK). Standardized 70/15/15 stratified split (Train: 16,947 | Val: 3,632 | Test: 3,632). 
* **Filters Applied:** Average (5x5 mean), Gaussian (5x5), Median (5x5), Sharpening (unsharp masking), and Sobel (edge detection, 3-channel replicated). Evaluated against a "No Filter" baseline (18 total runs).
* **Training Pipeline:** Uniform configuration across all runs to ensure fair comparison. 1 Warmup epoch (frozen backbone, LR `1e-3`) followed by 4 Fine-tuning epochs (unfrozen backbone, LR `1e-4`). Optimizer: Adam. Loss: Class-weighted. Batch Size: 64.

## 📊 Performance Results (Baseline vs. Filtered)

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Balanced Acc. (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | **No Filter** | **79.57** | **70.25** | **75.18** | **72.35** | **75.18** | **94.77** |
| ResNet50 | Average | 74.94 | 64.89 | 72.93 | 67.88 | 72.93 | 93.95 |
| ResNet50 | Gaussian | 76.27 | 66.75 | 75.93 | 70.07 | 75.93 | 94.58 |
| ResNet50 | Median | 77.42 | 69.27 | 73.28 | 70.95 | 73.28 | 94.56 |
| ResNet50 | Sharpening | 80.04 | 71.40 | 77.07 | 73.66 | 77.07 | 95.40 |
| ResNet50 | Sobel | 67.51 | 55.93 | 62.31 | 57.71 | 62.31 | 89.13 |
| **EfficientNet-B0** | **No Filter** | **76.38** | **66.90** | **75.80** | **70.09** | **75.80** | **94.55** |
| EfficientNet-B0| Average | 75.72 | 64.92 | 71.00 | 67.23 | 71.00 | 93.08 |
| EfficientNet-B0| Gaussian | 75.96 | 65.72 | 73.64 | 68.60 | 73.64 | 93.57 |
| EfficientNet-B0| Median | 72.96 | 62.74 | 71.33 | 65.48 | 71.33 | 93.00 |
| EfficientNet-B0| Sharpening | 79.93 | 71.15 | 76.43 | 73.43 | 76.43 | 95.14 |
| EfficientNet-B0| Sobel | 67.35 | 55.97 | 64.26 | 58.10 | 64.26 | 89.13 |
| **DenseNet121** | **No Filter** | **75.55** | **66.14** | **76.26** | **69.39** | **76.26** | **94.19** |
| DenseNet121 | Average | 74.56 | 64.35 | 71.98 | 67.30 | 71.98 | 92.64 |
| DenseNet121 | Gaussian | 78.28 | 69.03 | 74.02 | 70.22 | 74.02 | 94.26 |
| DenseNet121 | Median | 74.61 | 64.11 | 73.90 | 66.82 | 73.90 | 93.24 |
| DenseNet121 | Sharpening | 77.84 | 68.03 | 75.86 | 69.60 | 75.86 | 94.24 |
| DenseNet121 | Sobel | 63.99 | 54.30 | 64.56 | 55.82 | 64.56 | 88.49 |

*(Note: Full visualization assets including training curves, confusion matrices, ROC plots, and filter example grids are generated during execution and saved externally).*

## 📈 Comparative Analysis (Variance from Baseline)

| Model | Filter | Δ Accuracy | Δ Macro-F1 | Δ Balanced Acc. | Δ AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | Average | -4.63 | -4.47 | -2.25 | -0.82 |
| | Gaussian | -3.30 | -2.28 | 0.75 | -0.19 |
| | Median | -2.15 | -1.40 | -1.90 | -0.21 |
| | Sharpening | +0.47 | +1.31 | +1.89 | +0.63 |
| | Sobel | -12.06 | -14.64 | -12.87 | -5.64 |
| **EfficientNet-B0**| Average | -0.66 | -2.86 | -4.80 | -1.47 |
| | Gaussian | -0.42 | -1.49 | -2.16 | -0.98 |
| | Median | -3.42 | -4.61 | -4.47 | -1.55 |
| | Sharpening | +3.55 | +3.34 | +0.63 | +0.59 |
| | Sobel | -9.03 | -11.99 | -11.54 | -5.42 |
| **DenseNet121** | Average | -0.99 | -2.09 | -4.28 | -1.55 |
| | Gaussian | +2.73 | +0.83 | -2.24 | +0.07 |
| | Median | -0.94 | -2.57 | -2.36 | -0.95 |
| | Sharpening | +2.29 | +0.21 | -0.40 | +0.05 |
| | Sobel | -11.56 | -13.57 | -11.70 | -5.70 |

* **Mean Absolute Accuracy Impact:** Sobel (10.88) > Median (2.17) > Gaussian (2.15) > Sharpening (2.10) > Average (2.09).
* **Per-Class F1 Sensitivity:** AK (22.72) > BKL (17.92) > BCC (16.31) > MEL (15.26) > NV (7.92).

## 🔬 Technical Insights

* **Filter Impact Dynamics:** Smoothing filters generally degrade performance across all architectures. Sharpening marginally improves metrics, while Sobel edge detection causes a catastrophic drop in accuracy.
* **Architecture-Specific Behavior:** While 4 out of 5 filters behave consistently across all networks, the Gaussian filter shows architectural dependency (improving DenseNet121 by +2.73% but hurting ResNet50 and EfficientNet-B0).
* **Information Loss vs. CNN Feature Extraction:** Deep learning models inherently extract hierarchical features. Applying destructive hand-crafted filters (like Sobel, which discards color and texture for edge maps) actively works against the network's ability to learn rich representations. Conversely, Sharpening retains frequency content while boosting contrast, aligning better with CNN feature extraction.
* **Class Imbalance Sensitivity:** The least represented class (AK - 3.6% of dataset) showed the highest sensitivity to filter-induced distribution shifts (22.72 F1 swing). The most robust class was the majority class (NV - 53.2%).

## 🛠️ Reproduction Guide

1. Launch `ISIC2019_SkinLesion_Task02_FilterComparison.ipynb` in Google Colab (GPU runtime required).
2. Execute all cells sequentially. The dataset handles its own download via `kagglehub` (no API key required).
3. Checkpoints are automatically synced to Google Drive. In case of interruption, re-running the notebook will automatically detect and skip already completed filter/model iterations.

