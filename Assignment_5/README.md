# Assignment 5: Perceptron vs Multilayer Perceptron (MLP)

This repository contains a comparative analysis of a custom-built **Single-Layer Perceptron (PLA)** and a fully tuned **Multilayer Perceptron (MLP)** for multi-class image classification.

## 🎯 Objective
- Implement a **Perceptron Learning Algorithm (PLA)** from scratch using a One-vs-Rest (OvR) strategy.
- Implement and optimize a **Multilayer Perceptron (MLP)** for non-linear classification.
- Evaluate the impact of network depth, activation functions, and optimizers on model convergence.

---

## 📊 Dataset: English Handwritten Characters
A dataset featuring significant variability in stroke thickness, scale, and slant.

- **Total Samples:** 3,410 images.
- **Classes:** 62 unique categories (0–9, A–Z, a–z).
- **Preprocessing:** Resized to 32x32, Grayscale conversion, Flattening, and Normalization [0, 1].

---

## 🔬 Methodology

### 1. Single-Layer Perceptron (PLA)
- **Architecture:** Linear classifier implemented from scratch.
- **Strategy:** One-vs-Rest (OvR) for multi-class handling.
- **Limitations:** Limited to linearly separable patterns.

### 2. Multilayer Perceptron (MLP)
- **Architecture:** Feed-forward neural network implemented via `scikit-learn`.
- **Tuning:**
    - **Hidden Layers:** [1, 2, 3] layers (e.g., 512, 256).
    - **Activations:** ReLU vs. Tanh.
    - **Optimizers:** SGD vs. Adam.
- **Key Finding:** MLP vastly outperforms PLA by capturing non-linear relationships in image data.

---

## 📁 Repository Structure
The full analysis and implementation are documented in the notebook.

- **Notebook:** [experiment5.ipynb](./experiment5.ipynb)
- **Visualizations:** `Images/PNG/` and `Images/EPS/`.
    - Confusion Matrix
    - ROC Curves
    - Loss Curves

---

## 🛠️ Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
*Author: Monesh M*