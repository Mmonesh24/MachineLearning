# Assignment 2: Binary Classification using Naive Bayes and KNN

This repository contains a comparative study of **Naive Bayes** and **K-Nearest Neighbors (KNN)** algorithms for binary classification on the **Spambase** dataset.

## 🎯 Objective
- Implement Probabilistic Classifiers: **Gaussian**, **Multinomial**, and **Bernoulli Naive Bayes**.
- Implement **K-Nearest Neighbors (KNN)** with hyperparameter tuning for 'k'.
- Compare models based on Accuracy, Confusion Matrices, and ROC Curves.

---

## 📊 Dataset: Spambase
A dataset designed for spam email detection.

- **Features:** Word and character frequencies.
- **Target:** Spam (1) vs. Non-Spam (0).
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Spambase).

---

## 🔬 Methodology

### 1. Naive Bayes Classifiers
- **Gaussian NB:** Assumed normal distribution of features.
- **Multinomial NB:** Suitable for count/frequency data (word counts).
- **Bernoulli NB:** Suitable for binary/boolean features.

### 2. K-Nearest Neighbors (KNN)
- **Hyperparameter Tuning:** Optimized the number of neighbors ($k$) to balance bias and variance.
- **Distance Metric:** Euclidean distance.

### 3. Evaluation
- **ROC-AUC:** Analyzed the trade-off between True Positive Rate and False Positive Rate.
- **Conclusion:** Comparison of parametric (NB) vs. non-parametric (KNN) approaches.

---

## 📁 Repository Structure
- **Notebook:** [experiment2.ipynb](./experiment2.ipynb)
- **Dataset:** `spambase_csv.csv`

---

## 🛠️ Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
*Author: Monesh M*
