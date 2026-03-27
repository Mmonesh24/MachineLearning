# 🤖 Machine Learning Laboratory – Complete Assignment Portfolio

**Sri Sivasubramaniya Nadar College of Engineering, Chennai**
*(An autonomous Institution affiliated to Anna University)*

| Field | Details |
|---|---|
| **Name** | Monesh M |
| **Roll No** | 3122235001084 |
| **Degree & Branch** | BE Computer Science & Engineering |
| **Semester** | VI |
| **Subject** | Machine Learning Laboratory |
| **Academic Year** | 2025–2026 |
| **Batch** | 2023–2027 |

---

## 📋 Table of Contents

1. [Experiment 1 – Exploratory Data Analysis & Preprocessing](#-experiment-1--exploratory-data-analysis--preprocessing)
2. [Experiment 2 – Naive Bayes & KNN Classification](#-experiment-2--naive-bayes--knn-classification)
3. [Experiment 3 – Regression Analysis & Regularization](#-experiment-3--regression-analysis--regularization)
4. [Experiment 4 – Logistic Regression & SVM](#-experiment-4--logistic-regression--svm)
5. [Experiment 5 – Perceptron vs Multilayer Perceptron (MLP)](#-experiment-5--perceptron-vs-multilayer-perceptron-mlp)
6. [Experiment 6 – Decision Tree vs Random Forest](#-experiment-6--decision-tree-vs-random-forest)
7. [Experiment 7 – Bagging, Boosting & Stacked Ensembles](#-experiment-7--bagging-boosting--stacked-ensembles)
8. [Experiment 8 – Effect of PCA on ML Models](#-experiment-8--effect-of-pca-on-ml-models)
9. [Experiment 9 – Clustering Human Activity Recognition Data](#-experiment-9--clustering-human-activity-recognition-data)
10. [Global Technology Stack](#-global-technology-stack)

---

## 🔬 Experiment 1 – Exploratory Data Analysis & Preprocessing

📁 **Folder:** [`Assignment_1/`](./Assignment_1/)
📓 **Notebook:** [`assignment1_updated.ipynb`](./Assignment_1/assignment1_updated.ipynb)

### 🎯 Objective
To perform extensive **Exploratory Data Analysis (EDA)** and preprocessing across multiple heterogeneous datasets, building foundational data pipelines for subsequent ML experiments.

Students learned to:
- Handle missing values, detect outliers, and normalize features.
- Analyze data distributions, correlations, and class imbalances.
- Implement initial classification and regression baselines.

### 📊 Datasets Used

| Dataset | Purpose | Samples | Key Features |
|---|---|---|---|
| `loan_data.csv` | Loan approval prediction | ~600 | Credit Score, Income, Loan Amount |
| `DiabetesPredictionDataset.csv` | Medical risk classification | ~100K | Age, BMI, HbA1c, Blood Glucose |
| `email.csv` | Spam/ham email detection | ~5500 | Word & character frequency features |
| `iris.csv` | Multi-class flower classification | 150 | Sepal/Petal length & width |
| `TamilHandwrittenDataset.csv` | Character recognition | ~1000+ | Pixel intensity values |

### 🔍 EDA Highlights

#### Univariate Analysis
- **Histograms & KDE Plots**: Revealed that `loan_amount` and `HbA1c_level` are right-skewed, while `BMI` follows an approximately normal distribution.
- **Box Plots**: Detected outliers in `blood_glucose_level` (values > 300 mg/dL), requiring Winsorization.
- **Bar Charts**: Class imbalance discovered in spam detection (60% non-spam, 40% spam).

#### Bivariate & Multivariate Analysis
- **Scatter Plots**: Strong positive correlation between `Income` and `Loan Request Amount`.
- **Correlation Heatmaps**: High multicollinearity detected between `Average_Science` and `Total_Science` in academic data.
- **Pair Plots**: Iris dataset showed clear linear separability between *setosa* and other species in petal dimensions.

#### Key EDA Findings
- Missing values in `loan_data.csv` imputed with median for numerical and mode for categorical fields.
- Diabetes dataset features required standard scaling due to vastly different units (blood glucose vs. age).
- Iris is the only dataset with no missing values or significant outliers.

### ⚙️ Methodology
1. **Data Cleaning**: Missing value imputation, duplicate removal, outlier clipping.
2. **Encoding**: Label encoding for ordinal categories; One-Hot encoding for nominal categories.
3. **Scaling**: `StandardScaler` and `MinMaxScaler` evaluated for compatibility with different algorithms.
4. **Baseline Modeling**: Logistic Regression and Decision Tree as initial classifiers.

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations and array processing |
| `matplotlib` | Histograms, box plots, scatter plots |
| `seaborn` | Heatmaps, pair plots, distribution plots |
| `scikit-learn` | Preprocessing, baseline models, metrics |

---

## 🔬 Experiment 2 – Naive Bayes & KNN Classification

📁 **Folder:** [`Assignment_2/`](./Assignment_2/)
📓 **Notebook:** [`experiment2.ipynb`](./Assignment_2/experiment2.ipynb)

### 🎯 Objective
To implement and compare **probabilistic (Naive Bayes)** and **instance-based (KNN)** classifiers for binary spam detection, and evaluate the trade-offs between parametric vs. non-parametric approaches.

### 📊 Dataset: Spambase (UCI ML Repository)

| Property | Value |
|---|---|
| **Samples** | 4,601 |
| **Features** | 57 (word & character frequencies, capital run lengths) |
| **Target** | Spam (1) / Non-Spam (0) |
| **Class Distribution** | ~39% Spam, ~61% Non-Spam |
| **Source** | [UCI Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase) |

### 🔍 EDA Highlights

#### Distribution Analysis
- **Feature Histograms**: Most word-frequency features are heavily zero-inflated (sparse), making Bernoulli NB a natural candidate.
- **Capital Run Length**: Features like `capital_run_length_average` showed a heavy right tail, with spam emails having significantly longer capital sequences.
- **Class Balance Plot**: Moderate class imbalance (~60/40 split) – acceptable for direct accuracy metrics.

#### Correlation Analysis
- **Heatmap**: Found low inter-feature correlation in general, but word pairs like `remove` + `free` often co-occurred in spam messages.
- **Box Plots per Class**: Features like `word_freq_money`, `word_freq_free` showed dramatically higher values in spam (class=1).

#### Key EDA Findings
- Zero-inflated distributions favor **Bernoulli NB** and **KNN** over Gaussian NB.
- No missing values detected; dataset is clean and ready for direct modeling.
- Feature scaling is critical for KNN since Euclidean distance is scale-sensitive.

### ⚙️ Methodology
1. **Gaussian Naive Bayes**: Assumes normal feature distributions.
2. **Multinomial NB**: Suitable for count/frequency-based features.
3. **Bernoulli NB**: Optimal for binary/boolean feature presence.
4. **KNN Hyperparameter Tuning**: Tested `k = 1 to 30`, selected optimal `k` using cross-validation elbow.
5. **Evaluation**: Confusion matrix, Accuracy, Precision, Recall, F1, ROC-AUC.

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and inspection |
| `numpy` | Array operations |
| `matplotlib` | ROC curves, accuracy vs K plots |
| `seaborn` | Heatmaps, class distribution charts |
| `scikit-learn` | GaussianNB, MultinomialNB, BernoulliNB, KNeighborsClassifier, metrics |

---

## 🔬 Experiment 3 – Regression Analysis & Regularization

📁 **Folder:** [`Assignment_3/`](./Assignment_3/)
📓 **Notebook:** [`experiment3.ipynb`](./Assignment_3/experiment3.ipynb)

### 🎯 Objective
To model continuous financial/property values using **Linear Regression** and regularization techniques (**Ridge**, **Lasso**, **Elastic Net**), and study the bias-variance trade-off.

### 📊 Dataset: Loan/Property Analytics

| Property | Value |
|---|---|
| **Features** | Income, Property Price, Credit Score, Loan Term, etc. |
| **Target** | Loan Request Amount / Property Value |
| **Key Challenge** | Right-skewed targets requiring log-transformation |

### 🔍 EDA Highlights

#### Distribution Analysis
- **Target Variable Skewness**: `Loan Request Amount` strongly right-skewed (skewness > 2.0); log-transformation applied to approximate normality.
- **Income Distribution**: Multi-modal distribution suggesting distinct income brackets in the population.
- **Credit Score**: Approximately normally distributed with a mean ~680.

#### Correlation & Feature Analysis
- **Heatmap**: Strong positive correlation (r ≈ 0.78) between `Income` and `Loan Request Amount`.
- **VIF Analysis**: Detected multicollinearity between `Income` and `Property_Price`, prompting Ridge regularization.
- **Scatter Plots**: `Credit_Score` shows non-linear relationship with the target — modeled with polynomial features.

#### Key EDA Findings
- Log-transformation of skewed targets significantly improved R² (from 0.61 to 0.79).
- Multicollinearity between financial features validated the need for L2 regularization.
- Outliers in property prices removed via IQR-based clipping.

### ⚙️ Methodology
1. **OLS Linear Regression**: Minimize Residual Sum of Squares (RSS).
2. **Ridge (L2)**: Penalizes large coefficients to handle multicollinearity.
3. **Lasso (L1)**: Performs feature selection by driving some coefficients to zero.
4. **Elastic Net**: Combines L1 + L2 penalties for the best of both worlds.
5. **Alpha Tuning**: `GridSearchCV` used to find optimal regularization strength.
6. **Metrics**: R², RMSE, Mean Absolute Error (MAE).

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data ingestion, transformation |
| `numpy` | Log transformations, array operations |
| `matplotlib` | Residual plots, prediction vs actual plots |
| `seaborn` | Correlation heatmaps, distribution plots |
| `scikit-learn` | LinearRegression, Ridge, Lasso, ElasticNet, GridSearchCV, metrics |

---

## 🔬 Experiment 4 – Logistic Regression & SVM

📁 **Folder:** [`Assignment_4/`](./Assignment_4/)
📓 **Notebook:** [`experiment4.ipynb`](./Assignment_4/experiment4.ipynb)

### 🎯 Objective
To implement and compare **Logistic Regression** and **Support Vector Machines (SVM)** with multiple kernels for binary classification, optimizing decision boundaries on the high-dimensional Spambase dataset.

### 📊 Dataset: Spambase (UCI ML Repository)

| Property | Value |
|---|---|
| **Samples** | 4,601 |
| **Features** | 57 numerical attributes |
| **Target** | Spam (1) / Non-Spam (0) |
| **Split** | 80% Train / 20% Test |

### 🔍 EDA Highlights

#### Class Distribution
- **Bar Chart**: ~39% spam, ~61% non-spam. Slight imbalance managed without oversampling given the dataset size.

#### Feature Importance & Correlation
- **Correlation Heatmap**: Most features are weakly correlated with each other, validating the independence assumption of Naive Bayes from Experiment 2.
- **Top Discriminative Features**: `word_freq_remove`, `word_freq_free`, `char_freq_!` consistently showed highest correlation with the spam label.
- **Box Plots by Class**: Clear inter-class separation in frequency features for words like "free", "money", and "credit".

#### Key EDA Findings
- StandardScaler applied before SVM due to sensitivity to feature magnitudes.
- Features with very low variance (< 0.01) removed to reduce noise.
- PCA exploration showed ~95% variance retained in 30 components, validating feature redundancy.

### ⚙️ Methodology
1. **Logistic Regression**: Linear decision boundary, tuned `C` and `solver` via GridSearchCV.
2. **SVM (Linear Kernel)**: Large-margin linear classifier.
3. **SVM (RBF Kernel)**: Non-linear kernel mapping to higher dimensions.
4. **SVM (Polynomial Kernel)**: Captures polynomial interaction terms.
5. **SVM (Sigmoid Kernel)**: Mimics neural network behavior.
6. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, Learning Curves.

#### Visualizations Generated
- `class_distribution.png` – Bar chart of spam vs non-spam
- `correlation_heatmap.png` – Feature correlation matrix
- `confusion_matrices.png` – Side-by-side comparison of models
- `roc_curves.png` – ROC curves for Logistic Regression and all SVM kernels
- `learning_curves.png` – Training vs validation score by sample size

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and feature inspection |
| `numpy` | Matrix operations |
| `matplotlib` | ROC curves, confusion matrices, learning curves |
| `seaborn` | Heatmaps, box plots |
| `scikit-learn` | LogisticRegression, SVC, GridSearchCV, StandardScaler, metrics |

---

## 🔬 Experiment 5 – Perceptron vs Multilayer Perceptron (MLP)

📁 **Folder:** [`Assignment_5/`](./Assignment_5/)
📓 **Notebook:** [`experiment5.ipynb`](./Assignment_5/experiment5.ipynb)

### 🎯 Objective
To implement a **Perceptron Learning Algorithm (PLA)** from scratch and compare it with a deeply-tuned **Multilayer Perceptron (MLP)** for multi-class handwritten character recognition.

### 📊 Dataset: English Handwritten Characters

| Property | Value |
|---|---|
| **Samples** | 3,410 images |
| **Classes** | 62 (digits 0–9, uppercase A–Z, lowercase a–z) |
| **Image Size** | 32×32 pixels (grayscale) |
| **Feature Vector** | 1,024 (flattened pixels, normalized to [0,1]) |

### 🔍 EDA Highlights

#### Dataset Structure
- **Class Distribution Bar Chart**: Balanced dataset with approximately 55 samples per class.
- **Sample Images Grid**: Showed high variability in stroke thickness, scale, orientation, and slant across classes.
- **Pixel Intensity Distribution**: Most pixels are near 0 (background) or 1 (ink), forming a bimodal distribution.

#### Feature Analysis
- **PCA Visualization (2D)**: First two principal components showed moderate cluster separation for digits vs letters, but significant overlap within character subgroups.
- **Mean Image per Class**: Revealed characteristic "average" stroke patterns for each character, useful for understanding feature importance.
- **Variance Map**: Pixels at stroke-intersection regions showed highest variance across samples (informative regions).

#### Key EDA Findings
- Normalization to [0,1] was essential for both PLA and MLP convergence.
- 62-class problem significantly harder than binary; One-vs-Rest strategy chosen for PLA.
- Images exhibit high intra-class variance, motivating the need for deep, non-linear architectures.

### ⚙️ Methodology
1. **Perceptron (PLA)**: Implemented from scratch; One-vs-Rest (OvR) for multi-class.
2. **MLP Architecture Tuning**:
   - Hidden Layers: `[512]`, `[512, 256]`, `[512, 256, 128]`
   - Activations: `relu` vs `tanh`
   - Optimizers: `sgd` vs `adam`
3. **Training**: 80-20 split; early stopping applied via `max_iter`.
4. **Evaluation**: Accuracy, Confusion Matrix, ROC Curves (macro-OvR), Loss Curves.

#### Key Result
> MLP with 2 hidden layers [512, 256], ReLU activation, and Adam optimizer achieved ~4× higher accuracy than the single-layer Perceptron.

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn Pillow
```

| Library | Purpose |
|---|---|
| `pandas` | Data structuring |
| `numpy` | Image pixel flattening, normalization |
| `matplotlib` | Loss curves, confusion matrices, sample image grids |
| `seaborn` | Heatmaps, distribution plots |
| `scikit-learn` | MLPClassifier, Perceptron, metrics, PCA |
| `Pillow` | Image loading and preprocessing |

---

## 🔬 Experiment 6 – Decision Tree vs Random Forest

📁 **Folder:** [`Assignment_6/`](./Assignment_6/)
📓 **Notebook:** [`experiment6.ipynb`](./Assignment_6/experiment6.ipynb)

### 🎯 Objective
To implement and compare a single **Decision Tree (DT)** with a **Random Forest (RF)** ensemble on the Wisconsin Diagnostic Breast Cancer dataset, demonstrating the power of bagging for variance reduction.

### 📊 Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

| Property | Value |
|---|---|
| **Samples** | 569 |
| **Features** | 30 (Mean, Std. Error, Worst values of 10 cell-nucleus measurements) |
| **Target** | Malignant (M) / Benign (B) |
| **Class Distribution** | 37.3% Malignant, 62.7% Benign |
| **Source** | [UCI WDBC Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |

### 🔍 EDA Highlights

#### Class Distribution
- **Bar/Pie Chart**: Dataset is moderately imbalanced (212 malignant vs 357 benign). F1-score used as primary metric to account for this.

#### Feature Analysis & Correlation
- **Heatmap**: Strong positive correlation (r > 0.9) observed between `radius_mean` / `perimeter_mean` / `area_mean` — three features measure the same geometric property at different scales.
- **Box Plots by Diagnosis**: All "worst" features (`radius_worst`, `texture_worst`) showed clear separation between malignant and benign cases.
- **Violin Plots**: `concavity_mean` and `concave points_mean` showed bimodal distributions in malignant class, suggesting heterogeneous tumor morphology.

#### Outlier Analysis
- Several samples had extreme `area_worst` values (>3000 µm²) → confirmed malignant cases (clinically plausible, retained).

#### Key EDA Findings
- High multicollinearity among geometric features suggests Random Forest's feature subsampling is highly beneficial here.
- Feature importance analysis post-training confirmed `concave_points_worst` as the single most discriminative feature.

### ⚙️ Methodology
1. **Decision Tree**: Tuned via `GridSearchCV` (5-Fold CV) over `criterion`, `max_depth`, `min_samples_split`.
2. **Random Forest**: Bagging ensemble; tuned `n_estimators`, `max_features`.
3. **Feature Importance**: Plotted Gini importance from Random Forest.
4. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, Cross-validation scores.

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Dataset loading and EDA |
| `numpy` | Statistical calculations |
| `matplotlib` | Tree visualization, feature importance bar charts |
| `seaborn` | Heatmaps, violin plots, box plots |
| `scikit-learn` | DecisionTreeClassifier, RandomForestClassifier, GridSearchCV, metrics |

---

## 🔬 Experiment 7 – Bagging, Boosting & Stacked Ensembles

📁 **Folder:** [`Assignment_7/`](./Assignment_7/)
📓 **Notebook:** [`experiment7.ipynb`](./Assignment_7/experiment7.ipynb)

### 🎯 Objective
To implement and compare advanced **Ensemble Learning** strategies — **Bagging**, **Boosting (AdaBoost & Gradient Boosting)**, and **Stacking** — on the WDBC dataset to study bias-variance decomposition.

### 📊 Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

*(Same dataset as Experiment 6 — 569 samples, 30 features, binary target)*

### 🔍 EDA Highlights

*(Building on Exp 6 EDA)*

#### Error Analysis Across Models
- **Learning Curves**: Decision Tree shows classic high-variance behavior (large gap between train and validation score). Bagging shrinks this gap significantly.
- **Error Distribution Analysis**: Boosting models concentrate errors on the hardest-to-classify borderline tumors, revealing ambiguous cases.

#### Feature Interaction
- **Partial Dependence Plots (PDPs)**: `concave_points_worst` and `radius_worst` exhibit non-linear interaction — captured by Gradient Boosting but not linear models.
- **SHAP-style Feature Ordering**: Top 5 features across all ensemble models consistently include `concave_points_worst`, `perimeter_worst`, `radius_worst`.

#### Key EDA Findings
- Residual analysis showed that 8 borderline samples are consistently misclassified by single models but correctly classified by the Stacked Ensemble.
- Stacking successfully leveraged diverse predictions from SVM, Naive Bayes, and Decision Tree base learners.

### ⚙️ Methodology

| Model | Strategy | Goal |
|---|---|---|
| BaggingClassifier | Bootstrap Aggregation | Reduce variance |
| AdaBoostClassifier | Sequential weighting | Reduce bias |
| GradientBoostingClassifier | Residual minimization | Reduce bias + variance |
| StackingClassifier | Meta-learning (LR on top) | Exploit model diversity |

- **Meta-Learner**: Logistic Regression trained on out-of-fold predictions.
- **Evaluation**: 5-Fold CV Accuracy, F1, ROC-AUC, Learning Curves.

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data handling |
| `numpy` | Array computations |
| `matplotlib` | Learning curves, ROC plots |
| `seaborn` | Heatmaps, performance comparison plots |
| `scikit-learn` | BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, cross_val_score |

---

## 🔬 Experiment 8 – Effect of PCA on ML Models

📁 **Folder:** [`Assignment_8/`](./Assignment_8/)
📓 **Notebook:** [`Assignment_8.ipynb`](./Assignment_8/Assignment_8.ipynb)

### 🎯 Objective
To study the impact of **Principal Component Analysis (PCA)** on both regression and classification models by comparing performance **with and without** dimensionality reduction, using 5-fold cross-validation.

### 📊 Dataset: PCA Full Lab Dataset

| Property | Value |
|---|---|
| **Samples** | 1,000 |
| **Features** | 11 (Academic and performance-related metrics) |
| **Regression Target** | `Final_Score` (continuous) |
| **Classification Target** | `Performance_Level` (Low / Medium / High) |
| **Special Feature** | `Random_Noise_Feature` – intentional noise column |

### 🔍 EDA Highlights

#### Dataset Overview
- **Class Distribution (Classification)**: Performance levels are roughly balanced across Low, Medium, and High categories.
- **Target Distribution (Regression)**: `Final_Score` is approximately normally distributed (mean ~65, std ~15).

#### Feature Correlation & Redundancy
- **Heatmap**: High correlation detected between `Total_Science` and `Average_Science` (r ≈ 0.95), and between `Attendance` and `Participation` (r ≈ 0.87) — these redundant features are ideal candidates for PCA compression.
- **Scatter Plot Matrix**: Visual clustering confirms three distinct academic performance groups in the data.

#### PCA Analysis
- **Scree Plot** (`images/PNG/scree_plot.png`): Elbow at 7 components, capturing **95.90%** of total variance.
  - Components selected: 7 (from original 11)
- **Variance Explained Bar Chart**: First 3 principal components alone explain > 70% of variance.
- **PCA Biplot**: `Random_Noise_Feature` loads almost entirely on a component that explains <1% variance — effectively eliminated by PCA.

#### Key EDA Findings
- PCA successfully separates the `Random_Noise_Feature`, confirming that the dimensionality reduction is removing genuine noise.
- Standardization before PCA is critical: without it, high-magnitude features (like `Total_Score`) dominate the principal components.

### ⚙️ Methodology

**Preprocessing Pipeline:**
1. Label Encoding for categorical target (`Performance_Level`).
2. `StandardScaler` applied to all features (mandatory before PCA).
3. PCA fitted on training fold only (to prevent data leakage in 5-Fold CV).

**Models Evaluated:**

| Task | Model | Strategy |
|---|---|---|
| Regression | Linear Regression | Without PCA vs. With PCA (7 components) |
| Regression | Random Forest Regressor | Without PCA vs. With PCA |
| Classification | Logistic Regression | Without PCA vs. With PCA |
| Classification | Support Vector Machine (SVM) | Without PCA vs. With PCA |

### 📈 Results Summary

**Regression (5-Fold CV):**

| Model | Metric | No PCA | With PCA | Change |
|---|---|---|---|---|
| Linear Regression | MSE | 26.23 | 26.23 | ≈ Same |
| Linear Regression | R² | 0.748 | 0.748 | ≈ Same |
| Random Forest | MSE | 28.92 | 28.76 | ✅ Improved |
| Random Forest | R² | 0.723 | 0.724 | ✅ Improved |

**Classification (5-Fold CV):**

| Model | Metric | No PCA | With PCA | Change |
|---|---|---|---|---|
| Logistic Regression | Accuracy | 0.734 | 0.742 | ✅ Improved |
| Logistic Regression | F1-Score | 0.732 | 0.740 | ✅ Improved |
| SVM | Accuracy | 0.743 | 0.730 | ❌ Reduced |
| SVM | F1-Score | 0.737 | 0.722 | ❌ Reduced |

> **Key Insight:** Linear models benefit from PCA's orthogonalization, while SVM with RBF kernel can exploit original feature geometry better than the compressed representation.

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` | Scree plots, variance explained charts |
| `seaborn` | Heatmaps, scatter matrices |
| `scikit-learn` | PCA, LinearRegression, RandomForestRegressor, LogisticRegression, SVC, KFold, cross_val_score, StandardScaler, LabelEncoder |

---

## 🔬 Experiment 9 – Clustering Human Activity Recognition Data

📁 **Folder:** [`Assignment_9/`](./Assignment_9/)
📓 **Notebook:** [`Assignment_9.ipynb`](./Assignment_9/Assignment_9.ipynb)

### 🎯 Objective
To implement and compare unsupervised clustering algorithms — **K-Means**, **DBSCAN**, and **Hierarchical Agglomerative Clustering (HAC)** — on the Human Activity Recognition dataset, and evaluate their ability to recover true activity labels without supervision.

### 📊 Dataset: UCI Human Activity Recognition (HAR)

| Property | Value |
|---|---|
| **Source** | [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| **Subjects** | 30 volunteers, ages 19–48 |
| **Device** | Samsung Galaxy S II (waist-mounted) |
| **Samples** | 10,299 (train + test combined) |
| **Features** | 561 (time & frequency domain variables from accelerometer & gyroscope) |
| **True Labels** | 6 activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying |

### 🔍 EDA Highlights

#### Activity Distribution
- **Bar Chart** (`activity_distribution.png`): Reasonably balanced across the 6 activities (~1400–1900 samples each). No significant class imbalance.

#### Dimensionality & Feature Space
- **PCA Projection** (`pca_ground_truth.png`): After reducing 561 features to 2D via PCA, the ground-truth labels reveal:
  - Dynamic activities (Walking, Upstairs, Downstairs) form one broad cluster.
  - Static activities (Sitting, Standing, Laying) form another.
  - **The biggest challenge**: Sitting and Standing are nearly indistinguishable in 2D PCA space.
- **Variance Captured**: First 2 PCs capture ~65% of total variance; first 10 PCs capture ~85%.

#### Clustering Tendency Analysis
- **Silhouette Score vs K** (`kmeans_analysis.png`): Peak silhouette at k=2 (dynamic vs static split), with a local maximum at k=6.
- **WCSS Elbow Plot**: Clear elbow at k=6, aligning with the true number of activities.
- **Dendrogram** (`hierarchical_dendrogram.png`): Ward's linkage clearly shows two primary super-clusters (dynamic vs static), with each splitting further at finer scales.

#### Key EDA Findings
- The high-dimensionality (561 features) makes direct visualization impossible without PCA.
- DBSCAN with default parameters identifies most points as a single cluster, requiring careful parameter tuning (`eps=15`, `min_samples=10`).
- Stationary activities (Sitting vs Standing) are the primary source of error for all algorithms — a fundamental limitation of the sensor placement and feature set.

### ⚙️ Methodology

| Algorithm | Parameters | Rationale |
|---|---|---|
| K-Means | k=6, init='k-means++' | Number of true activity classes as guide |
| DBSCAN | eps=15, min_samples=10 | Tuned via k-distance plot |
| HAC (Ward's) | n_clusters=6, linkage='ward' | Minimizes intra-cluster variance |

**Preprocessing:**
1. Combined train + test sets.
2. Applied `StandardScaler` (mandatory for distance-based algorithms).
3. Used PCA (2D) for visualization only — clustering performed on full 561-D space.

### 📈 Performance Metrics

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI ↑ | NMI ↑ |
|---|---|---|---|---|---|
| K-Means | 0.110 | — | — | 0.420 | 0.559 |
| DBSCAN | 0.119 | — | — | 0.080 | 0.154 |
| **Hierarchical (HAC)** | **0.117** | — | — | **0.460** | **0.601** |

> **Best Performer:** Hierarchical Clustering (Ward's Linkage) — highest ARI (0.46) and NMI (0.60).

> **DBSCAN Limitation:** The uniform density of high-dimensional sensor data makes it difficult for DBSCAN to separate activities effectively — it found far fewer meaningful clusters.

### 📦 Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading and combining train/test splits |
| `numpy` | Numerical operations |
| `matplotlib` | Elbow plots, silhouette charts, dendrogram |
| `seaborn` | Activity distribution charts, scatter plots |
| `scikit-learn` | KMeans, DBSCAN, AgglomerativeClustering, PCA, StandardScaler, adjusted_rand_score, normalized_mutual_info_score, silhouette_score |
| `scipy` | `dendrogram`, `linkage` from `scipy.cluster.hierarchy` |

---

## 🛠️ Global Technology Stack

All experiments are implemented with a consistent Python-based environment:

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Environment** | Jupyter Notebook / JupyterLab |
| **Data Handling** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn` |
| **Deep Learning / NN** | `scikit-learn` (MLPClassifier) |
| **Dimensionality Reduction** | `scikit-learn` (PCA) |
| **Statistical Analysis** | `scipy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Image Processing** | `Pillow` (Exp 5 only) |
| **Reporting** | LaTeX, PDF |

### 🚀 Setup & Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Assignment

# 2. Install all dependencies at once
pip install pandas numpy matplotlib seaborn scikit-learn scipy Pillow

# 3. Launch Jupyter
jupyter notebook
```

### 📁 Complete Repository Structure

```
Assignment/
├── README.md                          ← This file
├── Assignment_1/
│   ├── assignment1_updated.ipynb
│   ├── DiabetesPredictionDataset.csv
│   ├── iris.csv, loan_data.csv, email.csv
│   ├── image_eps/
│   └── report1.pdf
├── Assignment_2/
│   ├── experiment2.ipynb
│   ├── spambase_csv.csv
│   └── report2.pdf
├── Assignment_3/
│   ├── experiment3.ipynb
│   ├── test.csv
│   └── Report3.pdf
├── Assignment_4/
│   ├── experiment4.ipynb
│   ├── spambase_csv.csv
│   ├── images/
│   └── Experiment_4_Binary_Classification_Report.pdf
├── Assignment_5/
│   ├── experiment5.ipynb
│   ├── Dataset/
│   ├── Images/
│   └── report5.pdf
├── Assignment_6/
│   ├── experiment6.ipynb
│   ├── Dataset/
│   ├── Images/
│   └── report6.pdf
├── Assignment_7/
│   ├── experiment7.ipynb
│   ├── Dataset/
│   ├── Images/
│   └── report7.pdf
├── Assignment_8/
│   ├── Assignment_8.ipynb
│   ├── Dataset.csv
│   ├── images/               ← scree_plot.png, etc.
│   ├── reg_results.csv
│   ├── clf_results.csv
│   └── Assignment_8.pdf
└── Assignment_9/
    ├── Assignment_9.ipynb
    ├── human+activity+recognition+using+smartphones/
    ├── images/               ← activity_distribution.png, pca_ground_truth.png, etc.
    ├── clustering_results.csv
    └── Assignment_9.pdf
```

---

## 📊 Experiments Summary Table

| # | Experiment | Dataset | Algorithm(s) | Task | Key Metric |
|---|---|---|---|---|---|
| 1 | EDA & Preprocessing | Loan, Diabetes, Iris, Spam, Tamil | — | EDA | Insight Quality |
| 2 | Naive Bayes & KNN | Spambase | GaussianNB, BernoulliNB, KNN | Classification | ROC-AUC |
| 3 | Regression & Regularization | Loan/Property | OLS, Ridge, Lasso, ElasticNet | Regression | R², RMSE |
| 4 | Logistic Regression & SVM | Spambase | LR, SVM (Linear/RBF/Poly) | Classification | F1, ROC-AUC |
| 5 | Perceptron vs MLP | Handwritten Characters | PLA (OvR), MLPClassifier | Classification | Accuracy |
| 6 | Decision Tree vs Random Forest | WDBC | DT, RF | Classification | F1, CV Score |
| 7 | Ensemble Methods | WDBC | Bagging, AdaBoost, GBM, Stacking | Classification | F1, ROC-AUC |
| 8 | PCA Impact Study | Academic Performance | PCA + LR, RF, SVM, LR | Reg + Clf | R², Accuracy |
| 9 | Clustering (Unsupervised) | UCI HAR | K-Means, DBSCAN, HAC | Clustering | ARI, NMI |

---

*Prepared as part of the 6th Semester Machine Learning Laboratory — Academic Year 2025–2026*
*SSN College of Engineering, Chennai.*
