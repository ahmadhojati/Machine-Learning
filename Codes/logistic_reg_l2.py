#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def logistic_reg_l2(X, Y, lrn_rte, num_itr, C):
    """
    Logistic Regression with L2 Regularization.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : numpy.ndarray
        Target vector of shape (n_samples,).
    lrn_rte : float
        Learning rate for gradient ascent.
    num_itr : int
        Number of iterations for gradient ascent optimization.
    C : float
        Regularization parameter (lambda).

    Returns:
    --------
    theta : numpy.ndarray
        Coefficients after logistic regression with L2 regularization.
    """

    # Add a column of ones to the feature matrix
    X = np.hstack((np.ones((len(X), 1)), X))
    theta = np.zeros((X.shape[1], 1))

    # Loop over the number of iterations
    for itr in range(num_itr):
        # Predict probability for each row in the dataset
        predictions = 1 / (1 + np.exp(-np.dot(X, theta)))

        # Calculate the errors
        errors = Y.reshape(-1, 1) - predictions

        # Loop over each weight coefficient
        for i in range(len(theta)):
            # Derivation of log-likelihood function + L2 regularization
            drvt = np.dot(errors.T, X[:, i])
            
            # Apply L2 regularization, except for the bias term (i.e., theta[0])
            if i != 0:
                drvt -= 2 * C * theta[i]

            # Gradient Ascent update
            theta[i] += lrn_rte * drvt

    return theta

