# Assignment 5: Perceptron vs Multilayer Perceptron (MLP)

This repository contains the implementation and comparative analysis of a custom-built Single-Layer Perceptron (PLA) and a fully tuned Multilayer Perceptron (MLP) for multi-class image classification. 

## 🎯 Objective
The primary goal of this assignment is to evaluate the performance differences between linear and non-linear classifiers on unstructured image data. Specifically, we:
1. Implement a **Perceptron Learning Algorithm (PLA)** from scratch, adapting it for 62-class classification using a One-vs-Rest (OvR) strategy.
2. Implement and rigorously tune a **Multilayer Perceptron (MLP)**, exploring the effects of network depth, activation functions, and optimizers on convergence and accuracy.

---

## 📊 Dataset Description
We utilize the **English Handwritten Characters Dataset**, which features significant variability in stroke thickness, scale, and slant.

* **Total Samples:** 3,410 images
* **Classes:** 62 unique categories (0–9, A–Z, a–z)
* **Distribution:** Perfectly balanced across all classes
* **Structure:** * `Dataset/Img/` - Contains the raw image files.
    * `Dataset/english.csv` - Contains the mapping of image filenames to their respective labels.

### Preprocessing Pipeline
To prepare the raw images for the neural networks, the following steps are applied:
* **Resizing:** Uniformly scaled to 32x32 pixels.
* **Grayscale Conversion:** Color channels dropped to reduce dimensionality.
* **Flattening:** Matrices converted into 1,024-dimensional feature vectors.
* **Normalization:** Pixel intensities scaled to a [0, 1] range to ensure stable gradient descent.

---



## 🧠 Methodology & Models

### Model A: Single-Layer Perceptron (PLA)
* Built entirely from scratch.
* Utilizes a **One-vs-Rest (OvR)** approach to handle the 62 distinct classes, training 62 independent binary classifiers.
* Serves as our linear baseline.

### Model B: Multilayer Perceptron (MLP)
* Implemented using `scikit-learn`.
* Subjected to extensive hyperparameter tuning:
    * **Hidden Layers:** Tested configurations from 1 up to 3 layers (e.g., 512, 256, 128).
    * **Activations:** Compared `ReLU` (faster convergence) vs. `Tanh` (better accuracy in deeper networks).
    * **Optimizers:** Evaluated `SGD` vs. `Adam`.
* **Key Finding:** The tuned MLP vastly outperforms the PLA. The PLA's linear nature makes it unsuitable for complex handwriting patterns, whereas the MLP successfully learns robust, non-linear representations through backpropagation.

---

## 📁 Repository Structure

```text
├── Dataset/                  # Directory containing raw images and labels
│   ├── Img/                  # Raw handwritten character images
│   └── english.csv           # Image-to-label mapping
├── Images/                   # Generated evaluation plots and EDA
│   ├── PNG/                  # Standard image formats (e.g., Confusion Matrix, ROC, Loss)
│   └── EPS/                  # High-resolution vector graphics for LaTeX reporting
├── experiment5.ipynb         # Main Jupyter Notebook containing all code and analysis
└── README.md                 # Project documentation