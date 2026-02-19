# Assignment 4: Binary Classification using Logistic Regression and SVM

This assignment involves implementing and comparing two major machine learning algorithms—Logistic Regression and Support Vector Machines (SVM)—for the task of binary classification on the Spambase dataset.

## Code
- `experiment4.ipynb`: Python notebook containing data preprocessing, exploratory data analysis, model training, hyperparameter tuning using Grid Search, and performance evaluation.

## Dataset
- `spambase_csv.csv`: A dataset consisting of 4601 email instances, each described by 57 features (frequencies of specific words and characters) and a target label (1 for spam, 0 for non-spam).

## How to Run
1. Ensure you have Python installed with the following libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
2. Open `experiment4.ipynb` in a Jupyter environment (VS Code, JupyterLab, etc.).
3. Run the cells sequentially to:
   - Load and explore the dataset.
   - Preprocess the data (scaling and splitting).
   - Train baseline and optimized Logistic Regression models.
   - Compare different SVM kernels (Linear, Poly, RBF, Sigmoid).
   - Evaluate model performance using Accuracy, F1-score, Confusion Matrices, and ROC curves.

## Visualizations
The experiment generates several plots saved in the `images/` directory, including:
- Class distribution analysis.
- Feature correlation maps.
- Confusion matrices for both models.
- ROC curves and AUC comparison.
- Learning curves to assess model generalization.
