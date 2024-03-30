Breast Cancer Detection Project
===============================

Overview
--------
This project aims to build machine learning models for the detection of breast cancer based on clinical and biopsied features. The dataset used in this project contains features computed from digitized images of fine needle aspirate (FNA) of breast masses, and the target variable indicates whether the mass is benign or malignant.

Dataset
-------
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, commonly referred to as the "WDBC dataset". It is publicly available on the UCI Machine Learning Repository and contains features computed from digitized images of FNA of breast masses.

Features:
- Mean, standard error, and worst (mean of the three largest values) of the following ten features computed for each cell nucleus:
  1. Radius
  2. Texture
  3. Perimeter
  4. Area
  5. Smoothness
  6. Compactness
  7. Concavity
  8. Concave points
  9. Symmetry
  10. Fractal dimension

Target Variable:
- Diagnosis (M = malignant, B = benign)

Project Structure
-----------------
- `Breast Cancer Classidication using Machine Learning.ipynb`: Jupyter Notebook containing the code for data exploration, preprocessing, model building, and evaluation.
- `README.md`: This file, providing an overview of the project.
- `data.csv`: File containing the dataset files.

Dependencies
------------
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Getting Started
---------------
1. Clone this repository:

    ```bash
    https://github.com/PlanetDestroyyer/Breast-Cancer-Classidication-using-Machine-Learning.git
    ```

2. Open and run the `Breast Cancer Classidication using Machine Learning.ipynb` notebook in Jupyter or any compatible environment.

Model Evaluation
----------------
In this project, we trained and evaluated several machine learning models for breast cancer detection, including:

- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes

Model performance was evaluated using accuracy score on both training and testing datasets.

