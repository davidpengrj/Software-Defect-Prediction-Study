# Software Defect Prediction Framework & Empirical Study on Borderline Oversampling

## 1. Project Overview

This project establishes a comprehensive empirical research framework for **Software Defect Prediction (SDP)**. It consists of two core components:
1.  **Foundation Framework**: Based on the paper *Software defect prediction: future directions and challenges*, implemented to explore the general SDP process.
2.  **Improved Framework**: Addressing the pervasive **Class Imbalance (Skewed Distribution)** issue in software engineering datasets, this framework introduces the **Borderline-SMOTE** technique, inspired by the paper *"Estimating Uncertainty in Line-Level Defect Prediction via Perceptual Borderline Oversampling"*.

### Core Highlights

* **Algorithm Diversity**
    This project implements **over 15 algorithms**, spanning from classic machine learning to deep learning:
    * **Ensemble Methods**: Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging, XGBoost.
    * **Deep Learning**: **Deep MLP** (3-layer structure: 100-50-25) to simulate deep feature extraction.
    * **Linear & Statistical Models**: Logistic Regression, SGD, Passive Aggressive, LDA, QDA.
    * **Classic Algorithms**: SVM, Naive Bayes, KNN, Decision Tree.

* **Robust Evaluation**
    Abandoning the misleading "Accuracy" metric (to avoid the "Accuracy Trap" in imbalanced data), this project evaluates models using **MCC (Matthews Correlation Coefficient)**, **ROC-AUC**, and **F1-Score**.

* **Methodological Innovation**
    Introduction of **Borderline Oversampling** to actively handle data skewness. Unlike random oversampling, this method synthesizes new data only for difficult samples near the decision boundary, enhancing the model's sensitivity to defects.

* **Datasets**
    Validated at the function/module level using 5 classic PROMISE datasets: `cm1`, `jm1`, `kc1`, `kc2`, `pc1`.

---

## 2. Repository Structure

This repository contains code frameworks representing three distinct stages of the study, designed for A/B testing:

### A. Core Contrastive Experiment (Paper-Inspired Experiment)
These two files form the core experiment to validate the effectiveness of the proposed method.

* **`SDP_Paper_Inspired_Oversampling.py` (Proposed Method)**
    * **Core Technique**: Uses **Borderline-SMOTE** to oversample minority class samples specifically located at the decision boundary in the training set.
    * **Models**: Includes 9 advanced models (GradientBoosting, AdaBoost, DeepMLP, etc.).
    * **Goal**: To demonstrate that focusing on borderline samples improves MCC and Recall.

* **`SDP_Paper_Inspired_Baseline_NoSampling.py` (Baseline)**
    * **Technique**: Uses standard preprocessing **without** any oversampling techniques.
    * **Models**: Identical 9 models as above.
    * **Goal**: Serves as the control group. Experiments show high "Accuracy" but low MCC, proving the detriment of imbalanced data.

### B. Foundation Framework
* **`Advanced_SDP_Framework.py`** (or `v3_Silent`)
    * **Technique**: Uses `class_weight='balanced'` (Cost-Sensitive Learning) instead of sampling.
    * **Models**: Contains 7 classic models (RandomForest, SVM, XGBoost, etc.).
    * **Function**: Acts as the foundational validation framework, supporting GridSearchCV and silent output mode.

---

## 3. Methodology

### 3.1 Data Preprocessing
* **Automated Cleaning**: Handles `?` missing values (using median imputation) and non-numeric features.
* **Stratified Split**: Uses `Stratified Train-Test Split` to preserve the defect ratio in both training and testing sets.

### 3.2 Core Innovation: Borderline Oversampling
To address the skewed distribution of defect knowledge, we adopted **Borderline-SMOTE**. This algorithm categorizes minority samples based on their K-Nearest Neighbors:
1.  **Safe**: Surrounded by the same class (Easy to classify).
2.  **Noise**: Surrounded by the opposite class (Likely outliers).
3.  **Danger (Borderline)**: Surrounded by mixed classes (The region of interest).

Our framework **only generates synthetic data for "Danger" samples**, forcing classifiers (like Boosting and MLP) to learn sharper decision boundaries.

---

## 4. Quick Start

### Prerequisites
It is recommended to use Anaconda. Install the required libraries:

```bash
# Create and activate environment
conda create -n sdp_env python=3.10
conda activate sdp_env

# Install dependencies (includes imbalanced-learn and xgboost)
pip install -r requirements.txt
