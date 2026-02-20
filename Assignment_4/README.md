# Assignment 4: Binary Classification using Logistic Regression and SVM

This repository contains a comparative study of **Logistic Regression** and **Support Vector Machines (SVM)** for binary classification tasks, applied to the **Spambase** dataset.

## 🎯 Objective
- Implement and optimize a **Logistic Regression** baseline model.
- Implement **Support Vector Machines (SVM)** with various kernels (Linear, Polymer, RBF, Sigmoid).
- Perform hyperparameter tuning using `GridSearchCV`.
- Compare models based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## 📊 Dataset: Spambase
The experiment utilizes the Spambase dataset to classify emails as spam or non-spam.

- **Total Samples:** 4601
- **Features:** 57 numerical attributes (word/character frequencies).
- **Target:** Spam (1) vs. Non-Spam (0).
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Spambase).

---

## 🔬 Methodology

### 1. Data Preprocessing
- **Scaling:** Applied `StandardScaler` to normalize feature distributions.
- **Splitting:** 80-20 Train-Test split.

### 2. Logistic Regression
- **Optimization:** Tuned `C` (regularization strength) and `solver`.
- **Goal:** Establish a linear baseline for classification performance.

### 3. Support Vector Machine (SVM)
- **Kernels Explored:** Linear, Polynomial, RBF, Sigmoid.
- **Tuning:** Optimized `C` (margin hardness) and `gamma` (kernel coefficient).
- **Observation:** proper kernel selection significantly impacts the decision boundary for high-dimensional data.

---

## 📁 Results & Visualizations
The models were evaluated using comprehensive metrics. Visualizations are stored in the `images/` directory.

- **Notebook:** [experiment4.ipynb](./experiment4.ipynb)
- **Visualizations:**
    - `class_distribution.png`
    - `correlation_heatmap.png`
    - `confusion_matrices.png`
    - `roc_curves.png`
    - `learning_curves.png`

---

## 🛠️ Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
*Author: Monesh M*
