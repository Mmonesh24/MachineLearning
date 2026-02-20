# Assignment 3: Regression Analysis

This repository contains an in-depth explororation of **Linear Regression** and regularization techniques to predict continuous target variables.

## 🎯 Objective
- Implement **Simple and Multiple Linear Regression**.
- Apply Regularization techniques: **Ridge (L2)**, **Lasso (L1)**, and **Elastic Net**.
- Analyze the impact of hyperparameter tuning on model complexity and overfitting.

---

## 📊 Dataset: Loan/Property Analytics
A dataset containing financial and property features.

- **Features:** Income, Property Price, Credit Score, etc.
- **Target:** Loan Request Amount / Property Value.
- **Preprocessing:** Log-transformation of skewed variables to satisfy normality assumptions.

---

## 🔬 Methodology

### 1. Linear Regression
- **Goal:** Minimize Residual Sum of Squares (RSS).
- **Assumption:** Linear relationship between independent and dependent variables.

### 2. Regularized Regression
- **Ridge:** Adds L2 penalty to shrink coefficients, reducing variance.
- **Lasso:** Adds L1 penalty for feature selection (driving some coefficients to zero).
- **Elastic Net:** Combines L1 and L2 penalties.

### 3. Evaluation
- **Metrics:** $R^2$ Score, Root Mean Squared Error (RMSE).
- **Analysis:** Comparison of training vs. testing error to detect overfitting.

---

## 📁 Repository Structure
- **Notebook:** [experiment3.ipynb](./experiment3.ipynb)
- **Dataset:** `test.csv`

---

## 🛠️ Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
*Author: Monesh M*
