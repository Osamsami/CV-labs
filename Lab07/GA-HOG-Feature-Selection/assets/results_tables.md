## Table 1

| Parameter                   | Baseline: HOG + SVM                                      | Proposed: GA-HOG + SVM                                   |
|:----------------------------|:---------------------------------------------------------|:---------------------------------------------------------|
| Medical dataset             | BloodMNIST (MedMNIST+, 128x128)                          | BloodMNIST (MedMNIST+, 128x128)                          |
| Image resolution            | 128 x 128                                                | 128 x 128                                                |
| HOG orientations            | 9                                                        | 9                                                        |
| Pixels per cell             | 8 x 8                                                    | 8 x 8                                                    |
| Cells per block             | 2 x 2                                                    | 2 x 2                                                    |
| Block normalization         | L2-Hys                                                   | L2-Hys                                                   |
| HOG feature dimension       | 8100                                                     | 8100                                                     |
| Feature selection           | None                                                     | Genetic Algorithm                                        |
| Features used by classifier | 8100                                                     | 2525                                                     |
| HOG implementation          | OpenCV HOGDescriptor                                     | OpenCV HOGDescriptor                                     |
| Classifier                  | SVM                                                      | SVM                                                      |
| SVM kernel                  | RBF (precomputed)                                        | RBF (precomputed)                                        |
| Hyperparameter tuning       | Validation set (C [1, 10, 100], gamma x [0.5, 1.0, 2.0]) | Validation set (C [1, 10, 100], gamma x [0.5, 1.0, 2.0]) |
| Selected C                  | 10.0                                                     | 10.0                                                     |
| Selected gamma multiplier   | 1.0                                                      | 1.0                                                      |
| Data split                  | Fixed (11959 / 1712 / 3421)                              | Fixed (11959 / 1712 / 3421)                              |
| Random seed                 | 42                                                       | 42                                                       |

## Table 2

| Metric                           |   HOG + SVM (Baseline) |   GA-HOG + SVM (Proposed) |   Improvement |
|:---------------------------------|-----------------------:|--------------------------:|--------------:|
| Accuracy (%)                     |                81.1166 |                   79.3335 |       -1.7831 |
| Precision (Macro, %)             |                81.7464 |                   80.6543 |       -1.0921 |
| Recall (Macro, %)                |                79.1591 |                   77.6554 |       -1.5038 |
| Macro-F1 (%)                     |                80.2297 |                   78.8666 |       -1.3631 |
| Weighted-F1 (%)                  |                80.9905 |                   79.2244 |       -1.7661 |
| Balanced accuracy (%)            |                79.1591 |                   77.6554 |       -1.5038 |
| Specificity (Macro, %)           |                97.2092 |                   96.9345 |       -0.2747 |
| Cohen's kappa                    |                 0.7785 |                    0.7575 |       -0.021  |
| Matthews correlation coefficient |                 0.7789 |                    0.758  |       -0.0209 |

## Table 3

| Item                        |   HOG + SVM (Baseline) |   GA-HOG + SVM (Proposed) |
|:----------------------------|-----------------------:|--------------------------:|
| Feature dimension           |               8100     |                  2525     |
| Feature reduction (%)       |                  0     |                    68.827 |
| HOG extraction time (s)     |                  7.365 |                     7.365 |
| GA selection time (s)       |                  0     |                   114.984 |
| SVM tuning time (s)         |                 31.16  |                    30.2   |
| Final SVM training time (s) |                  4.123 |                     3.665 |
| Test inference time (s)     |                  0.978 |                     0.741 |
| Total pipeline time (s)     |                 42.648 |                   156.213 |

## Table 4

| Setting   |   Population |   Generations |   Mutation prob. |   Crossover prob. |   Best fitness |   Val Macro-F1 (GA subset) |   Selected features |   Reduction (%) |   GA time (s) |
|:----------|-------------:|--------------:|-----------------:|------------------:|---------------:|---------------------------:|--------------------:|----------------:|--------------:|
| GA-1      |           20 |            20 |            0.005 |               0.8 |         0.7249 |                      0.756 |                2525 |         68.8272 |       114.984 |
| GA-2      |           30 |            20 |            0.02  |               0.8 |         0.7171 |                      0.755 |                3071 |         62.0864 |       190.372 |

## Per-Class Results

| Class                |   Baseline F1 |   GA-HOG F1 |   Support |   Difference |
|:---------------------|--------------:|------------:|----------:|-------------:|
| basophil             |        0.7366 |      0.7483 |       244 |       0.0117 |
| eosinophil           |        0.7951 |      0.7559 |       624 |      -0.0392 |
| erythroblast         |        0.8624 |      0.8548 |       311 |      -0.0076 |
| immature granulocyte |        0.6922 |      0.6783 |       579 |      -0.0139 |
| lymphocyte           |        0.8487 |      0.8468 |       243 |      -0.0019 |
| monocyte             |        0.642  |      0.616  |       284 |      -0.026  |
| neutrophil           |        0.8767 |      0.8488 |       666 |      -0.0279 |
| platelet             |        0.9646 |      0.9604 |       470 |      -0.0042 |

## Bonus 1

| Metric                           |   All HOG (8100) |   Random (2525) |   ANOVA F-test (2525) |   GA (2525) |
|:---------------------------------|-----------------:|----------------:|----------------------:|------------:|
| Accuracy (%)                     |          81.1166 |         79.1581 |               80.3566 |     79.3335 |
| Precision (Macro, %)             |          81.7464 |         80.1642 |               80.5025 |     80.6543 |
| Recall (Macro, %)                |          79.1591 |         77.3106 |               78.1846 |     77.6554 |
| Macro-F1 (%)                     |          80.2297 |         78.4382 |               79.0676 |     78.8666 |
| Weighted-F1 (%)                  |          80.9905 |         79.0292 |               80.1124 |     79.2244 |
| Balanced accuracy (%)            |          79.1591 |         77.3106 |               78.1846 |     77.6554 |
| Specificity (Macro, %)           |          97.2092 |         96.913  |               97.1017 |     96.9345 |
| Cohen's kappa                    |           0.7785 |          0.7554 |                0.7696 |      0.7575 |
| Matthews correlation coefficient |           0.7789 |          0.756  |                0.7701 |      0.758  |

