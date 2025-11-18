
# Software Defect Prediction Framework & Empirical Study on Borderline Oversampling

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Imbalanced-Learn](https://img.shields.io/badge/Technique-Borderline--SMOTE-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📖 Project Overview

This project establishes a comprehensive empirical research framework for **Software Defect Prediction (SDP)**. It addresses the critical challenge of class imbalance in software engineering datasets by integrating advanced oversampling techniques.

The project consists of two core components:

1. **Foundation Framework**: Based on the paper *Software defect prediction: future directions and challenges*, implemented to explore the general SDP process.
2. **Improved Framework**: Addressing the pervasive **Class Imbalance (Skewed Distribution)** issue. It introduces the **Borderline-SMOTE** technique, inspired by the paper *"Estimating Uncertainty in Line-Level Defect Prediction via Perceptual Borderline Oversampling"*.

---

## 🌟 Core Highlights

### 🧠 Algorithm Diversity

This project implements over **15 algorithms**, spanning from classic machine learning to deep learning architectures:

| Category                                   | Algorithms & Description                                                                                                           |
| :----------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **A. Ensemble Methods**<br>*(Most Robust)* | • **Random Forest**<br>• **Extra Trees**<br>• **Gradient Boosting**<br>• **AdaBoost**<br>• **XGBoost**<br>• **Bagging Classifier** |
| **B. Deep Learning**                       | • **MLPClassifier**<br>• **Deep MLP**                                                                                              |
| **C. Linear & Statistical**                | • **Logistic Regression**<br>• **SGD Classifier**<br>• **Passive Aggressive**<br>• **LDA & QDA**                                   |
| **D. Classic Algorithms**                  | • **SVC**<br>• **GaussianNB**<br>• **KNeighbors**<br>• **Decision Tree**                                                           |

---

## 📊 Robust Evaluation

> **⚠️ The Accuracy Trap**
> Accuracy is abandoned since it is misleading in imbalanced settings.

Primary evaluation metrics:

* **MCC**
* **ROC-AUC**
* **F1-Score**

---

## 💡 Methodological Innovation

The core of the improved method is **Borderline-SMOTE**, which generates synthetic data only for *“danger”* minority samples—those lying near class boundaries.

This enhances the model’s capability to detect actual defects.

---

## 📂 Datasets

Evaluated on 5 PROMISE datasets: `cm1`, `jm1`, `kc1`, `kc2`, `pc1`.

---

# 🏗️ Repository Structure

```
.
├── SDP_Paper_Inspired_Oversampling.py         # Proposed method (Borderline-SMOTE)
├── SDP_Paper_Inspired_Baseline_NoSampling.py  # Baseline without sampling
├── Advanced_SDP_Framework.py                  # Foundation cost-sensitive framework
├── requirements.txt
└── README.md
```

### Detailed Description

1. **`SDP_Paper_Inspired_Oversampling.py` (Proposed Method)**

   * Applies **Borderline-SMOTE**
   * Enhances MCC and Recall
   * Uses 9 advanced models

2. **`SDP_Paper_Inspired_Baseline_NoSampling.py` (Baseline)**

   * No oversampling
   * Demonstrates the **Accuracy Trap**

3. **`Advanced_SDP_Framework.py` (Foundation)**

   * Uses cost-sensitive learning (`class_weight='balanced'`)
   * Supports GridSearchCV

---

# ⚙️ Methodology

### 3.1 Data Preprocessing

* Handles missing values (`?`) using median imputation
* Stratified train-test split

### 3.2 Borderline Oversampling

Borderline-SMOTE classifies minority instances into:

* 🟢 Safe
* 🔴 Noise
* ⚠️ Danger (**oversampled**)

> The system generates synthetic samples only for **danger** points.

---

# 🚀 Quick Start

### Prerequisites

Use Anaconda:

```bash
# 1. Create environment
conda create -n sdp_env python=3.10
conda activate sdp_env

# 2. Install dependencies
pip install -r requirements.txt
```

---

# ▶️ Running the Experiments

## **1. Run the Improved Method (View Performance Gains)**

```bash
python SDP_Paper_Inspired_Oversampling.py
```

---

## **2. Run the Baseline (View Original Performance)**

```bash
python SDP_Paper_Inspired_Baseline_NoSampling.py
```

---

## **3. Run the Foundation Framework**

```bash
python Advanced_SDP_Framework.py
```

---

# 🧪 Key Findings

Through comparative experiments, we reached the following conclusions:

### 🚫 **The Accuracy Trap**

Baseline models (e.g., AdaBoost) reached **90% Accuracy** on `cm1`, but MCC was **0.0**, meaning:

* The classifier predicted **all samples as "No Defect"**
* It learned *nothing*

### ✅ **Effectiveness of Oversampling**

After introducing **Borderline-SMOTE**:

* Accuracy dropped slightly (back to reality)
* **MCC rose significantly** (e.g., 0.00 → 0.37+)
  This proves the model began truly identifying defects.

### 🏅 **Model Recommendations**

* **ExtraTrees** & **GradientBoosting** are most robust with oversampling
* **Deep MLP** captures non-linear patterns effectively

---

# 📚 References

1. Verma, A., & Sharma, A. (2024). *Software defect prediction: future directions and challenges.* Empirical Software Engineering, 29(6), 143.
2. Chen, W., et al. *Estimating Uncertainty in Line-Level Defect Prediction via Perceptual Borderline Oversampling.*


