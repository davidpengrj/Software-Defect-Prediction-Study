# Software Defect Prediction Framework & Empirical Study on Borderline Oversampling

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Imbalanced-Learn](https://img.shields.io/badge/Technique-Borderline--SMOTE-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📖 Project Overview

This project establishes a comprehensive empirical research framework for **Software Defect Prediction (SDP)**. It addresses the critical challenge of class imbalance in software engineering datasets by integrating advanced oversampling techniques.

The project consists of two core components:
1.  **Foundation Framework**: Based on the paper *Software defect prediction: future directions and challenges*, implemented to explore the general SDP process.
2.  **Improved Framework**: Addressing the pervasive **Class Imbalance (Skewed Distribution)** issue. It introduces the **Borderline-SMOTE** technique, inspired by the paper *"Estimating Uncertainty in Line-Level Defect Prediction via Perceptual Borderline Oversampling"*.

---

## 🌟 Core Highlights

### 🧠 Algorithm Diversity
This project implements over **15 algorithms**, spanning from classic machine learning to deep learning architectures:

| Category | Algorithms & Description |
| :--- | :--- |
| **A. Ensemble Methods**<br>*(Most Robust)* | • **Random Forest**: Bagging-based, reduces variance.<br>• **Extra Trees**: Extremely Randomized Trees, superior variance control.<br>• **Gradient Boosting**: Iteratively corrects errors of predecessors.<br>• **AdaBoost**: Focuses on misclassified samples.<br>• **XGBoost**: Optimized distributed gradient boosting.<br>• **Bagging Classifier**: General bootstrap aggregating. |
| **B. Deep Learning** | • **MLPClassifier**: Basic structure.<br>• **Deep MLP**: 3-layer structure (100-50-25) to simulate deep feature extraction. |
| **C. Linear & Statistical** | • **Logistic Regression**: Classic linear baseline.<br>• **SGD Classifier**: Stochastic Gradient Descent for large-scale data.<br>• **Passive Aggressive**: Online learning for data streams.<br>• **LDA & QDA**: Discriminant Analysis. |
| **D. Classic Algorithms** | • **SVC**: Support Vector Machine.<br>• **GaussianNB**: Naive Bayes.<br>• **KNeighbors**: KNN.<br>• **Decision Tree**: CART. |

### 📊 Robust Evaluation
> **⚠️ The Accuracy Trap**: We explicitly abandon the misleading "Accuracy" metric, which is prone to bias in imbalanced datasets.

Instead, we evaluate models using:
* **MCC (Matthews Correlation Coefficient)**: The most reliable metric for imbalanced binary classification.
* **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve.
* **F1-Score**: Harmonic mean of Precision and Recall.

### 💡 Methodological Innovation
Introduction of **Borderline Oversampling** to actively handle data skewness. Unlike random oversampling, this method synthesizes new data **only for difficult samples near the decision boundary**, enhancing the model's sensitivity to defects.

### 📂 Datasets
Validated at the function/module level using 5 classic **PROMISE datasets**: `cm1`, `jm1`, `kc1`, `kc2`, `pc1`.

---

## 🏗️ Repository Structure

This repository is designed for **A/B Testing** across three distinct stages:

```text
.
├── SDP_Paper_Inspired_Oversampling.py      # [Proposed Method] Core experiment with Borderline-SMOTE
├── SDP_Paper_Inspired_Baseline_NoSampling.py # [Baseline] Control group with no sampling
├── Advanced_SDP_Framework.py               # [Foundation] Cost-Sensitive learning framework
├── requirements.txt                        # Project dependencies
└── README.md
