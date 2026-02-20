# Assignment 6: Decision Tree vs. Random Forest

This repository contains a comparative study of **Decision Tree** and **Random Forest** algorithms for categorical classification using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

## 🎯 Objective
- Implement a **Decision Tree (DT)** and optimize its structure via 5-Fold Cross-Validation.
- Extend the DT into a **Random Forest (RF)** ensemble to observe variance reduction.
- Analyze the impact of tree depth, splitting criteria, and ensemble size on robustness.

---

## 📊 Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
A classic high-dimensional biomedical dataset for diagnostic prediction.

- **Total Samples:** 569
- **Features:** 30 numerical attributes (Mean, Std Error, Worst).
- **Target:** Malignant (M) vs. Benign (B).
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- **Class Distribution:** Evaluated target balance.
- **Feature Correlation:** Heatmaps to identify redundant features.

### 2. Decision Tree Implementation
- **Hyperparameters:** `criterion` (Gini/Entropy), `max_depth`, `min_samples_split`.
- **Tuning:** `GridSearchCV` with 5-Fold CV to prevent overfitting.

### 3. Random Forest (Ensemble)
- **Strategy:** Bagging (Bootstrap Aggregation).
- **Tuning:** `n_estimators`, `max_features`.
- **Observation:** The ensemble approach produces smoother decision boundaries and higher generalization.

---

## 📁 Results & Visualizations
Comparative metrics and plots are generated in the notebook.

- **Notebook:** [experiment6.ipynb](./experiment6.ipynb)
- **Visualizations:** `Images/PNG/` and `Images/EPS/`.
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
