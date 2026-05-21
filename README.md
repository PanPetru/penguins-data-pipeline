# Palmer Penguins Data Engineering & Classification Pipeline

An end-to-end data processing and machine learning pipeline focusing on data cleansing, statistical feature engineering, dimensionality reduction, and model validation using the Palmer Penguins dataset.

## Project Architecture

The project is structured into sequential stages, mirroring an enterprise data engineering lifecycle:

1. **`01_data_imputation.py`** Handles missing values in features. Evaluates and compares different imputation strategies including Mean, Median, Mode, and Conditional Group-Mean Imputation to ensure data consistency without losing valuable distribution insights.
   
2. **`02_statistical_analysis.py`** Performs detailed data exploration. Computes covariance and correlation matrices. Implements feature scaling transformations including Min-Max Scaling, Z-Score Standardization, and L1/L2 Row-wise Normalization.

3. **`03_classification_knn_svm.py`** Implements a multi-class classification pipeline. Evaluates model behavior by tuning hyperparameters for $k$-Nearest Neighbors (kNN) and comparing performance results against Support Vector Machines (SVM).

4. **`04_dimensionality_reduction.py`** Optimizes computational efficiency and visualizes data structures by reducing features from 4D space to 2D space. Implements and compares unsupervised **Principal Component Analysis (PCA)** and supervised **Linear Discriminant Analysis (LDA)**.

5. **`05_model_validation.py`** Ensures model robustness and prevents overfitting by executing a **5-Fold Cross-Validation** strategy, gathering accurate statistics on mean accuracy and model stability.

## Core Engineering Competencies Demonstrated
* **Data Quality Engineering:** Proficient in data cleansing, outlier management, and structural imputation.
* **Analytical Problem Solving:** Ability to evaluate metrics and make data-driven decisions regarding feature manipulation.
* **Algorithmic Comprehension:** Understanding how underlying data distributions directly impact mathematical models (kNN, SVM, PCA, LDA).
