# Machine Learning Research & Implementation Portfolio

A technical repository documenting systematic research into statistical learning algorithms, deep neural architectures, and feature engineering strategies. This portfolio spans from fundamental exploratory analysis to complex non-linear representation learning.

---

## 🔬 Core Research Domains

*   **Exploratory Data Synthesis:** High-dimensional data visualization, statistical characterization, and noise/missing-value mitigation.
*   **Probabilistic Learning:** Bayesian inference and distribution-based classification.
*   **Geometric Classifiers:** Large-margin optimization through SVMs and instance-based proximity modeling (KNN).
*   **Regularized Linear Dynamics:** L1/L2 regularization to mitigate overfitting in complex regression tasks.
*   **Neural Architectures:** Comparative analysis of single-layer Perceptrons and deep multi-layer neural networks using backpropagation.

---

## 📁 Laboratory Experiments

### [Exp 1: Data Synthesis & Exploratory Analysis](./Assignment_1/)
*   **Objective**: Comprehensive statistical characterization across diverse domains (Medical, Financial, NLP).
*   **Key Contributions**: Implementation of robust preprocessing pipelines, feature correlation mapping, and initial classification baselines.
*   **Datasets**: Loan Approval, Diabetes Risk, Spam Detection, Iris species, and Tamil digits.

### [Exp 2: Probabilistic & Instance-based Classification](./Assignment_2/)
*   **Objective**: Comparative analysis of Generative vs. Discriminative learning paradigms.
*   **Key Contributions**: Implementation of Gaussian, Multinomial, and Bernoulli Naive Bayes variants. Optimal $k$ determination through cross-validation in KNN classifiers.
*   **Metrics**: Sensitivity-Specificty trade-offs using ROC curves and AUC analysis.

### [Exp 3: Regularized Linear Dynamics & Regression](./Assignment_3/)
*   **Objective**: Modeling continuous loan request dynamics using regularized linear models.
*   **Key Contributions**: Comparative evaluation of **OLS**, **Ridge** (L2), **Lasso** (L1), and **Elastic Net**. Feature engineering via logarithmic transformations and scaling.
*   **Focus**: Bias-Variance trade-off optimization through alpha-parameter tuning.

### [Exp 4: Geometric Learning & Large Margin Classifiers](./Assignment_4/)
*   **Objective**: Optimizing decision boundaries using Logistic Regression and Support Vector Machines.
*   **Key Contributions**: Kernel analysis for SVM (Linear, RBF, Polynomial, Sigmoid). Optimization of hyperplanes for high-dimensional feature spaces.
*   **Discovery**: Demonstrated superiority of non-linear kernels in handling high-frequency NLP features.

### [Exp 5: Neural Architectures & Non-linear Representation](./Assignment_5/)
*   **Objective**: Evaluating the shift from linear Perceptrons to deep Multilayer Perceptrons (MLP).
*   **Key Contributions**: Custom implementation of Perceptron Learning Algorithm (OvR). Design and tuning of deep MLP architectures (500+ neurons, Tanh/ReLU activations, Adam optimizer).
*   **Results**: Achieved a 4x performance gain over linear models on unstructured handwritten character recognition.

### [Exp 6: Ensemble Dynamics & Tree-based Learning](./Assignment_6/)
*   **Objective**: Comparative study of single Decision Trees vs. Random Forest ensembles.
*   **Key Contributions**: Recursive feature splitting optimization, 5-Fold Cross-Validation for tree regularization. Implementation of bootstrap aggregation (Bagging) to minimize variance.
*   **Discovery**: Validated the "Wisdom of the Crowd" through RF, achieving superior generalization on high-dimensional biomedical data (WDBC).

---

## 🛠️ Technological Stack

*   **Runtime**: Python 3.10+
*   **Core Logic**: `numpy` (Tensor operations), `pandas` (Structured serialization).
*   **Analytics**: `scikit-learn` (Optimization & Evaluation), `scipy` (Statistical tests).
*   **Visualization**: `matplotlib`, `seaborn` (Professional plotting & Heatmaps).
*   **Document Engineering**: `LaTeX` (Report generation), `Jupyter` (Interactive research).

---

## 👋 About the Author
**Monesh M**  
*Computer Science & Engineering*  
*SSN College of Engineering*  

---
*Generated with pride as a reflection of 6th semester Machine Learning Laboratory work.*
