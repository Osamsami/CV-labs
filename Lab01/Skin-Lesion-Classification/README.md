# Skin Lesion Classification — ISIC 9-Class

A comparative computer vision experiment for multi-class skin lesion classification using transfer learning, deep feature extraction, classical machine learning classifiers, and computational efficiency analysis.

> This project is intended for educational and research purposes only and is not a medical diagnostic system.

## Overview

This lab investigates image classification on the ISIC 9-class skin lesion dataset.

The experiment consists of three main stages:

1. Transfer learning using multiple pretrained CNN architectures.
2. Deep feature extraction from the best-performing CNN followed by classical machine learning classification.
3. Computational efficiency comparison of the evaluated CNN architectures.

## Dataset

**Dataset:** Skin Cancer ISIC (9 classes)

**Source:** Kaggle

The dataset contains images belonging to 9 different skin lesion classes.

## Reference

Ratul et al., *Skin Lesions Classification Using Deep Learning Based on Dilated Convolution*, bioRxiv, 2020.

## Objectives

- Apply transfer learning to a multi-class image classification problem.
- Compare different pretrained CNN architectures.
- Evaluate classification performance using multiple metrics.
- Extract deep features from the best transfer learning model.
- Compare classical machine learning classifiers using the extracted features.
- Analyze the trade-off between accuracy and computational efficiency.

## Transfer Learning Models

The following pretrained CNN architectures were evaluated:

- AlexNet
- VGG16
- VGG19
- ResNet18
- ResNet50
- ResNet101
- DenseNet121
- EfficientNet-B0

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC

## Experiment 1 — Transfer Learning

The CNN architectures were fine-tuned on the ISIC 9-class dataset.

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|
| AlexNet | 49.15 | 45.88 | 49.31 | 42.83 | 88.68 |
| VGG16 | **61.02** | **62.06** | **59.03** | **57.33** | 89.52 |
| VGG19 | 49.15 | 55.53 | 49.31 | 42.47 | **90.79** |
| ResNet18 | 54.24 | 57.64 | 53.47 | 49.48 | 90.52 |
| ResNet50 | 53.39 | 57.94 | 52.78 | 51.14 | 89.82 |
| ResNet101 | 53.39 | 59.05 | 52.78 | 51.42 | 90.64 |
| DenseNet121 | 54.24 | 57.17 | 53.47 | 51.18 | 89.27 |
| EfficientNet-B0 | 53.39 | 53.05 | 52.78 | 51.18 | 90.04 |

### Best Transfer Learning Model

**VGG16**

VGG16 achieved the highest classification accuracy at **61.02%**.

It also achieved the highest precision and F1-score among the evaluated transfer learning models.

VGG19 achieved the highest AUC at **90.79%**, but VGG16 provided better overall classification performance based on accuracy and F1-score.

## Experiment 2 — Deep Features + Classical ML

Deep features were extracted from VGG16 and used as input to several classical machine learning classifiers.

The following classifiers were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Linear SVM
- RBF-SVM
- XGBoost

## Classical Classifier Results

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 51.69 | 57.14 | 51.39 | 48.42 | 90.10 |
| Decision Tree | 49.15 | 50.74 | 49.31 | 47.81 | 71.38 |
| Random Forest | 59.32 | 63.75 | 57.64 | 54.84 | 89.19 |
| K-Nearest Neighbors | 55.08 | 53.82 | 54.17 | 51.52 | 84.71 |
| Linear SVM | 54.24 | 53.52 | 53.47 | 49.49 | 90.47 |
| RBF-SVM | **62.71** | **63.89** | **60.42** | **58.63** | 90.32 |
| XGBoost | 58.47 | 60.64 | 56.94 | 54.83 | 87.57 |

### Best Classical Classifier

**RBF-SVM**

RBF-SVM achieved the highest accuracy of **62.71%**.

It also achieved the highest precision, recall, and F1-score among the evaluated classical classifiers.

## Experiment 3 — Computational Efficiency

The CNN architectures were compared based on computational requirements.

The following factors were analyzed:

- Number of parameters
- Model size
- FLOPs
- Inference time
- Accuracy

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| AlexNet | 57.041 | 217.6 | 0.374 | 0.897 | 49.15 |
| VGG16 | 134.297 | 512.316 | 7.95 | 6.014 | **61.02** |
| VGG19 | 139.607 | 532.573 | 10.073 | 7.678 | 49.15 |
| ResNet18 | 11.181 | 42.734 | 0.930 | 2.146 | 54.24 |
| ResNet50 | 23.526 | 90.061 | 2.108 | 8.281 | 53.39 |
| ResNet101 | 42.519 | 162.816 | 4.012 | 11.850 | 53.39 |
| DenseNet121 | 6.963 | 27.183 | 1.478 | 16.888 | 54.24 |
| EfficientNet-B0 | **4.019** | **15.715** | **0.211** | 9.306 | 53.39 |

## Key Findings

### VGG16 was the strongest transfer learning model

VGG16 achieved **61.02% accuracy**, making it the best-performing CNN according to the primary accuracy metric.

### RBF-SVM achieved the highest overall accuracy

Using deep features extracted from VGG16 with RBF-SVM resulted in **62.71% accuracy**, slightly outperforming direct VGG16 classification.

### Deep features can be combined with classical ML

The experiment demonstrated that high-level representations learned by a CNN can be used as input to traditional machine learning algorithms.

Among the evaluated classifiers, RBF-SVM performed best.

### Model efficiency varies significantly

VGG16 achieved the highest accuracy but required approximately:

- 134.30M parameters
- 512.32 MB model size
- 7.95 GFLOPs
- 6.01 ms average inference time

EfficientNet-B0 was considerably smaller, with approximately:

- 4.02M parameters
- 15.72 MB model size
- 0.211 GFLOPs

However, its accuracy was lower at 53.39%.

This demonstrates the trade-off between predictive performance and computational efficiency.

## Overall Results

| Experiment | Best Model | Accuracy |
|---|---|---:|
| Transfer Learning | VGG16 | **61.02%** |
| Deep Features + Classical ML | RBF-SVM | **62.71%** |
| Most Compact CNN | EfficientNet-B0 | 53.39% |

### Best Overall Pipeline

**VGG16 Deep Feature Extraction + RBF-SVM**

This combination achieved the highest accuracy in the experiments:

**62.71%**

## Workflow

```text
ISIC 9-Class Dataset
        |
        v
Image Preprocessing
        |
        v
Transfer Learning
        |
        v
Multiple CNN Architectures
        |
        v
Model Evaluation
        |
        v
Best CNN: VGG16
        |
        v
Deep Feature Extraction
        |
        v
Classical ML Classifiers
        |
        v
Best Classifier: RBF-SVM
        |
        v
62.71% Accuracy



Technologies Used

Python 

 PyTorch 

 TorchVision 

 Scikit-learn 

 XGBoost 

 NumPy 

 Pandas 

 THOP 

 Tabulate 

 Google Colab 

Notebook

The complete implementation, experiments, evaluation, and results are available in:



CV_SkinLesionClassification.ipynb

Disclaimer

This project is intended for educational and research purposes only.



The results should not be interpreted as medical advice, diagnosis, or a clinically validated diagnostic system.'''

