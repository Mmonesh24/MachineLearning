# Assignment 7: Bagging, Boosting, and Stacked Ensemble Models

This repository contains a comprehensive study of **Ensemble Learning strategies** applied to the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

## 🎯 Objective
- Implement and analyze **Bagging** (Bootstrap Aggregation) using Decision Trees.
- Implement and analyze **Boosting** methods (AdaBoost and Gradient Boosting).
- Build a **Stacked Ensemble** combining heterogeneous models (SVM, Naive Bayes, Decision Tree) with a Logistic Regression meta-learner.
- Evaluate the bias-variance tradeoff and generalization capabilities.

---

## 📊 Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
A classic high-dimensional biomedical dataset for diagnostic prediction.

- **Total Samples:** 569
- **Features:** 30 numerical attributes.
- **Target:** Malignant (M) vs. Benign (B).
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

---

## 🔬 Ensemble Strategies

### 1. Bagging (Bootstrap Aggregation)
- **Base Estimator:** Decision Tree.
- **Goal:** Reduce **variance** by averaging independent high-variance models.
- **Tuning:** `n_estimators`, `max_samples`.

### 2. Boosting
- **Algorithms:** AdaBoost, Gradient Boosting.
- **Goal:** Reduce **bias** by sequentially correcting errors of predecessors.
- **Tuning:** `learning_rate`, `n_estimators`.

### 3. Stacked Ensemble
- **Base Models:** SVM, Naive Bayes, Decision Tree.
- **Meta-Learner:** Logistic Regression.
- **Goal:** Combine diverse predictive patterns for superior performance.

---

## 📁 Results & Visualizations
The ensemble models achieved high accuracy and robustness.

- **Notebook:** [experiment7.ipynb](./experiment7.ipynb)
- **Visualizations:** `Images/PNG` and `Images/EPS`.
    - `class_distribution.png`
    - `correlation_heatmap.png`
    - `confusion_matrices.png`
    - `roc_comparison.png`

---

## 🛠️ Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
*Author: Monesh M*
