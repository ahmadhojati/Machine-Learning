#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd
import numpy as np

def inner_kfold_cross_validation(n_splits, X, y, param_grid):
    """
    Perform inner k-fold cross-validation for Support Vector Machines (SVM) with different kernels and parameters.

    Parameters:
    - n_splits (int): Number of folds for cross-validation.
    - X (array-like): Input features.
    - y (array-like): Target variable.
    - param_grid (list of tuples): List of parameter combinations for SVM.

    Returns:
    - pd.DataFrame: DataFrame containing mean validation accuracy for each set of parameters across inner folds.
    """

    # Defining a K-Fold cross-validation
    skf = KFold(n_splits=n_splits, random_state=1234, shuffle=True)

    # Train and validation data index
    ind = list(skf.split(X, y))

    # List to store results
    results = []

    # Loop over different parameter combinations
    for param_set in param_grid:
        # Creating different classifiers for each kernel
        clf_rbf = SVC(kernel='rbf', C=param_set[0], gamma=param_set[1])
        clf_poly = SVC(kernel='poly', C=param_set[0], gamma=param_set[1], degree=param_set[2])
        clf_linear = SVC(kernel='linear', C=param_set[0], gamma=param_set[1])

        # Model for transferring the data to polynomial
        poly = PolynomialFeatures(degree=int(param_set[2]))

        # List to store results for each inner fold
        fold_results = []

        # Loop over the folds
        for fold_idx in range(n_splits):
            # Train and validation data for each fold
            X_train = X[ind[fold_idx][0]]
            y_train = y[ind[fold_idx][0]]
            X_valid = X[ind[fold_idx][1]]
            y_valid = y[ind[fold_idx][1]]

            # Train and validation data for polynomial features
            X_poly_train = poly.fit_transform(X[ind[fold_idx][0]])
            X_poly_valid = poly.fit_transform(X[ind[fold_idx][1]])

            # Standardize the train and validation data
            sc = StandardScaler()
            X_train_std = sc.fit_transform(X_train)
            X_valid_std = sc.transform(X_valid)
            X_poly_train_std = sc.fit_transform(X_poly_train)
            X_poly_valid_std = sc.transform(X_poly_valid)

            # Fitting the classifiers
            clf_rbf.fit(X_train_std, y_train)
            clf_poly.fit(X_train_std, y_train)
            clf_linear.fit(X_poly_train_std, y_train)

            # Validation accuracy
            acc_rbf = clf_rbf.score(X_valid_std, y_valid)
            acc_poly = clf_poly.score(X_valid_std, y_valid)
            acc_linear = clf_linear.score(X_poly_valid_std, y_valid)

            # Appending the output of each inner fold
            fold_results.append([param_set[0], param_set[1], param_set[2], acc_rbf, acc_poly, acc_linear])

        # Calculate mean of the inner folds
        mean_result = np.mean(fold_results, axis=0)

        # Append parameters with validation accuracy to the results list
        results.append({'C': mean_result[0], 'gamma': mean_result[1], 'degree': mean_result[2],
                        'rbf': mean_result[3], 'poly': mean_result[4], 'linear': mean_result[5]})

    # Convert results list to DataFrame
    df = pd.DataFrame(results)

    return df

