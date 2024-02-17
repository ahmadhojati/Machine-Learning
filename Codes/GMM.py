#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import multivariate_normal

def GMM(X, p, mu1, mu2, sgma1, sgma2, nitr):
    """
    Gaussian Mixture Model (GMM) implementation.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data array of shape (n_samples, n_features).
    p : numpy.ndarray
        Initial mixing coefficients of the two Gaussian distributions.
    mu1 : numpy.ndarray
        Initial mean of the first Gaussian distribution.
    mu2 : numpy.ndarray
        Initial mean of the second Gaussian distribution.
    sgma1 : numpy.ndarray
        Initial covariance matrix of the first Gaussian distribution.
    sgma2 : numpy.ndarray
        Initial covariance matrix of the second Gaussian distribution.
    nitr : int
        Number of iterations for the expectation-maximization algorithm.

    Returns:
    --------
    p : numpy.ndarray
        Updated mixing coefficients of the two Gaussian distributions.
    mu1 : numpy.ndarray
        Updated mean of the first Gaussian distribution.
    mu2 : numpy.ndarray
        Updated mean of the second Gaussian distribution.
    sgma1 : numpy.ndarray
        Updated covariance matrix of the first Gaussian distribution.
    sgma2 : numpy.ndarray
        Updated covariance matrix of the second Gaussian distribution.
    log_likelihood : numpy.ndarray
        Log likelihood values over the iterations.
    """

    # Initialize arrays for responsibilities and log likelihood
    resp = np.zeros((len(X), 2))
    log_likelihood = np.zeros((nitr, 1))

    # Expectation-Maximization Algorithm
    for i in range(nitr):
        # Expectation Step
        f = p[0] * multivariate_normal(mu1, sgma1).pdf(X) + p[1] * multivariate_normal(mu2, sgma2).pdf(X)
        z1 = (p[0]) * multivariate_normal(mu1, sgma1).pdf(X) / f
        z2 = (p[1]) * multivariate_normal(mu2, sgma2).pdf(X) / f

        # Maximization Step
        p[0] = np.sum(z1) / len(X)
        p[1] = np.sum(z2) / len(X)
        mu1 = np.dot(z1.T, X) / np.sum(z1)
        mu2 = np.dot(z2.T, X) / np.sum(z2)
        sgma1 = np.dot(z1 * (X - mu1).T, (X - mu1)) / np.sum(z1)
        sgma2 = np.dot(z2 * (X - mu2).T, (X - mu2)) / np.sum(z2)

        # Update responsibilities
        resp[:, 0] = p[0] * multivariate_normal(mu1, sgma1).pdf(X)
        resp[:, 1] = p[1] * multivariate_normal(mu2, sgma2).pdf(X)

        # Compute log likelihood
        log_likelihood[i] = np.sum(np.log(np.sum(resp, axis=1)))

    return p, mu1, mu2, sgma1, sgma2, log_likelihood

