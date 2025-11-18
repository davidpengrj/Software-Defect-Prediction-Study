Empirical Study of Software Defect Prediction via Borderline Oversampling
1. Project Overview
This project is a comprehensive empirical study on Software Defect Prediction (SDP). It addresses the critical challenges of class imbalance (skewed distribution) and evaluation validity in software engineering datasets.

Based on recent literature feedback, specifically referencing "Estimating Uncertainty in Line-Level Defect Prediction via Perceptual Borderline Oversampling", this project moves beyond traditional methods by implementing Borderline-SMOTE to synthesize difficult-to-classify samples near the decision boundary.

Key Objectives (Addressing Feedback)
Algorithm Diversity: Implemented 15+ algorithms, ranging from classic ML (Random Forest, SVM) to advanced Ensemble Learning (Gradient Boosting, ExtraTrees) and Deep Learning (Deep MLP).

Robustness: Addressed the "Accuracy Trap" (where 90% accuracy can be misleading) by focusing on MCC (Matthews Correlation Coefficient) and ROC-AUC.

Methodological Innovation: Implemented Borderline Oversampling to actively handle data imbalance, responding to the limitation of "skewed distribution of defect knowledge."

Comprehensive Evaluation: Evaluated across 5 classic PROMISE datasets (cm1, jm1, kc1, kc2, pc1) at the function/module level.

2. Repository Structure
This repository contains three distinct frameworks representing different stages of the study:

A. The "Paper-Inspired" Experiment (A/B Test)
These two files form the core comparative experiment to validate the effectiveness of the proposed method.

SDP_Paper_Inspired_Oversampling.py (The Proposed Method)

Core Technique: Implements Borderline-SMOTE to oversample the minority class (defects) specifically near the decision boundary.

Models: Uses 9 advanced models including GradientBoosting, AdaBoost, ExtraTrees, DeepMLP (3-Layers), LDA, QDA, etc.

Goal: To demonstrate that focusing on borderline samples improves the model's ability to distinguish actual defects (Higher MCC/Recall).

SDP_Paper_Inspired_Baseline_NoSampling.py (The Baseline)

Technique: Uses standard preprocessing without any oversampling.

Models: Same 9 models as above.

Goal: To serve as a control group. It often achieves high "Accuracy" but low "MCC," proving the "Accuracy Trap" in imbalanced datasets.

B. The Foundation Framework
Advanced_SDP_Framework_v3_Silent.py

Technique: Uses class_weight='balanced' (cost-sensitive learning) instead of sampling.

Models: A different set of 7 classic models (RandomForest, SVM, XGBoost, LogisticRegression, etc.).

Output: Optimized for silent execution, producing summary tables for MCC, F1, and ROC-AUC at the end.

3. Methodology
3.1 Data Preprocessing
Datasets: PROMISE Software Engineering Repository (cm1, jm1, kc1, kc2, pc1).

Cleaning: Automated handling of missing values (median imputation) and '?' characters.

Stratified Split: 80/20 Train-Test split preserving the class distribution ratio.

3.2 The Core Innovation: Borderline Oversampling
Addressing the critique that standard evaluation is not comprehensive enough, we adopted the strategy from the reference paper Chen et al.. Instead of random oversampling, Borderline-SMOTE categorizes minority samples into:

Safe: Surrounded by minority class (Easy to classify).

Noise: Surrounded by majority class (Likely outliers).

Danger (Borderline): Surrounded by mixed classes.

Our framework only generates synthetic data for the "Danger" samples, forcing the classifiers (Boosters/Deep MLP) to learn sharper decision boundaries.

3.3 Model Construction
We employed GridSearchCV (Cross-Validation) for all models to optimize hyperparameters, ensuring fair comparison.

Deep Learning: A Multi-Layer Perceptron (MLP) with a 3-layer architecture (100-50-25 neurons) to capture non-linear features.

Ensemble: Extensive use of Boosting (Gradient, Ada) and Bagging (ExtraTrees) to reduce variance and bias.

4. How to Run
Prerequisites
Create a Python environment (e.g., using Anaconda) and install the required dependencies:

Bash

# Create environment
conda create -n sdp_env python=3.10
conda activate sdp_env

# Install dependencies
conda install -c conda-forge pandas scikit-learn xgboost imbalanced-learn tabulate
Running the Experiment
1. To see the improved performance (Proposed Method):

Bash

python SDP_Paper_Inspired_Oversampling.py
2. To see the baseline for comparison (Control Group):

Bash

python SDP_Paper_Inspired_Baseline_NoSampling.py
3. To run the standard weighted framework:

Bash

python Advanced_SDP_Framework_v3_Silent.py
5. Key Findings & Results
Our empirical results demonstrate a significant trade-off between Accuracy and true predictive power (MCC):

The Accuracy Trap: In the Baseline study (NoSampling), models like AdaBoost often achieved 90%+ Accuracy on datasets like cm1, but with an MCC of 0.0. This indicates the model was simply predicting "No Defect" for everything.

Effectiveness of Oversampling: In the Proposed study (Oversampling), Accuracy slightly decreased (e.g., to 75-80%), but MCC rose significantly (e.g., to 0.37+).

Best Models:

ExtraTrees and GradientBoosting consistently performed best when combined with Borderline Oversampling.

DeepMLP showed stable performance across varying datasets.

This confirms that Borderline Oversampling effectively addresses the skewed distribution problem, making the defect prediction models actionable and reliable.
