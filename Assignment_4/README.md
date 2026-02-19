# Assignment 4: Binary Classification using Linear and Kernel-Based Models

## Objective
The objective of this assignment is to perform binary classification on the Spambase dataset using Logistic Regression and Support Vector Machines (SVM). We also perform extensive Exploratory Data Analysis (EDA) and visualize model performance using ROC curves and learning curves.

## Dataset
- **Name**: Spambase Dataset
- **Description**: A collection of spam and non-spam emails. Features include word frequencies, character frequencies, and capital run lengths.
- **Target**: `class` (0: Non-spam/Ham, 1: Spam)

## Directory Structure
```
Assignment_4/
├── ml_ex4.ipynb        # Jupyter notebook with code
├── spambase_csv.csv    # Dataset
├── README.md           # Instructions and details
├── report.tex          # LaTeX report source
├── Report4.pdf         # Generated PDF report
└── image/              # Visualizations
    ├── EPS/            # EPS format plots
    └── PNG/            # PNG format plots
```

## How to Run
1. Ensure you have the required libraries installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
2. Open `ml_ex4.ipynb` in Jupyter Notebook or VS Code.
3. Run all cells to process the data, train models, and generate plots.

## Model Performance
- **Logistic Regression**: High accuracy and efficiency.
- **Support Vector Machine**: Excellent performance with RBF kernel.
- Visualizations for Confusion Matrix, ROC-AUC, and Learning Curves are available in the `image` folder.
