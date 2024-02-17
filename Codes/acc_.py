#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
import pandas as pd
import numpy as np

def acc_(n_splits, X, y, parm):
    """
    Inner K-Fold Cross Validation for different classifiers.

    Parameters:
    -----------
    n_splits : int
        Number of splits for K-Fold cross-validation.
    X : numpy.ndarray
        Input features array of shape (n_samples, n_features).
    y : numpy.ndarray
        Target array of shape (n_samples,).
    parm : list
        List of classifier parameters.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing mean accuracy for different classifiers and parameters.
    """

    # Defining a K-Fold cross-validation
    skf = KFold(n_splits=n_splits, random_state=1234, shuffle=True)

    # Train and validation data index
    ind = list(skf.split(X, y))

    # Empty DataFrame for results
    df = pd.DataFrame(columns=['C', 'gamma', 'degree', 'rbf', 'poly', 'linear'])

    # Loop over the length of the parameters and different folds
    for i in range(len(parm)):
        # Creating different classifiers
        clf_rbf = SVC(kernel='rbf', C=parm[i][0], gamma=parm[i][1])
        clf_poly = SVC(kernel='poly', C=parm[i][0], gamma=parm[i][1], degree=parm[i][2])
        clf_polyf = SVC(kernel='linear', C=parm[i][0], gamma=parm[i][1])

        # Model for transferring the data to polynomial
        poly = PolynomialFeatures(degree=(parm[i][2]).astype(int))

        # Empty DataFrame for results
        df1 = pd.DataFrame()

        # Loop over the folds
        for j in range(0, n_splits):
            # Train and validation data for each fold
            X_train = X[ind[j][0]]
            y_train = y[ind[j][0]]
            X_valid = X[ind[j][1]]
            y_valid = y[ind[j][1]]

            # Train and validation for polynomial features
            X_polyf_train = poly.fit_transform(X[ind[j][0]])
            X_polyf_valid = poly.fit_transform(X[ind[j][1]])

            # Standardize the train and validation data
            sc = StandardScaler()
            X_train_std = sc.fit_transform(X_train)
            X_valid_std = sc.transform(X_valid)
            X_polyf_train_std = sc.fit_transform(X_polyf_train)
            X_polyf_valid_std = sc.transform(X_polyf_valid)

            # Fitting the classifiers
            clf_rbf.fit(X_train_std, y_train)
            clf_poly.fit(X_train_std, y_train)
            clf_polyf.fit(X_polyf_train_std, y_train)

            # Validation accuracy
            acc_rbf = clf_rbf.score(X_valid_std, y_valid)
            acc_poly = clf_poly.score(X_valid_std, y_valid)
            acc_polyf = clf_polyf.score(X_polyf_valid_std, y_valid)

            # Appending the output of each inner fold
            df1 = df1.append(pd.DataFrame([parm[i][0], parm[i][1], parm[i][2], acc_rbf, acc_poly, acc_polyf]),
                             ignore_index=True)

        # Mean of the inner folds
        df2 = np.mean(df1, axis=1)

        # Parameters with validation accuracy
        df = df.append({'C': df2[0], 'gamma': df2[1], 'degree': df2[2], 'rbf': df2[3],
                        'poly': df2[4], 'linear': df2[5]},
                       ignore_index=True)

    return df

