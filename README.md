# Palmer Penguins - Data Processing & ML Practice

A simple Python project created to learn the basics of data preparation, exploratory analysis, and basic machine learning models using the popular Palmer Penguins dataset.

## How the project is structured:

I divided the project into 5 sequential scripts to practice different stages of working with data:

1. **`01_data_imputation.py`** – Handling missing data. I tested and compared different simple ways to fill in missing values (using mean, median, mode).

2. **`02_statistical_analysis.py`** – Basic data exploration. Calculating correlations and practicing feature scaling (Min-Max Scaling and Z-Score Standardization).

3. **`03_classification_knn_svm.py`** – Trying out basic classification algorithms. I implemented and tuned hyperparameters for $k$-Nearest Neighbors (kNN) and Support Vector Machines (SVM).

4. **`04_dimensionality_reduction.py`** – Practice with reducing features. I wanted to see how Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) can simplify 4D data into a 2D plot.

5. **`05_model_validation.py`** – Checking model reliability. Using 5-Fold Cross-Validation to see how stable my classification results actually are.

## What I practiced here:
* Using **Pandas** and **NumPy** for clean data manipulation and handling missing values.
* Implementing basic Machine Learning workflows using **scikit-learn**.
* Splitting code into clean, single-purpose scripts instead of keeping everything in one messy notebook.

## How to Run

1. Clone the repository and ensure `palmerpenguins_original.csv` is in the same directory as the scripts.
2. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib

3. Run any script in order, for example:

   ```bash
   python 01_data_imputation.py