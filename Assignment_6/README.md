# Assignment 6: Decision Tree vs. Random Forest

This repository contains a comparative study of **Decision Tree** and **Random Forest** algorithms for categorical classification using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

## 🎯 Objective
- Implement a **Decision Tree (DT)** and optimize its structure via 5-Fold Cross-Validation.
- Extend the DT into a **Random Forest (RF)** ensemble to observe the reduction in variance and improvement in generalization.
- Critically analyze the impact of tree depth, splitting criteria, and ensemble size on model robustness.

---

## 📊 Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
A classic high-dimensional biomedical dataset for diagnostic prediction.
- **Total Samples:** 569
- **Features:** 30 numerical attributes (Mean, Std Error, and "Worst" of 10 cell nucleus characteristics).
- **Target:** Malignant (M) vs. Benign (B).
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- **Class Distribution:** Evaluated potential skewness in the target variable.
- **Feature Correlation:** Generated heatmaps to identify redundant features that might influence tree splitting logic.

### 2. Decision Tree Implementation
- **Hyperparameter Space:** Explored `criterion` (Gini, Entropy), `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **Tuning:** Used `GridSearchCV` with 5-Fold CV to find the "Goldilocks" depth (sufficient complexity without overfitting).

### 3. Random Forest (Ensemble)
- **Extending DTs:** Built a Bagging-based ensemble using bootstrap aggregation.
- **Hyperparameter tuning:** Focused on `n_estimators`, `max_features`, and `bootstrap`.
- **Generalization:** Observed how the committee of trees produces a smoother decision boundary compared to a single volatile tree.

---

## 📁 Results & Implementation
The full analysis, visualization, and metric tables are documented in the Jupyter Notebook.

- **Notebook:** [experiment6.ipynb](./experiment6.ipynb)
- **Visualizations:**
    - `Images/PNG/`: Standard plots (ROC, Confusion Matrix).
    - `Images/EPS/`: Vector graphics for professional report scaling.

---

## 🛠️ Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
*Author: Monesh M*
