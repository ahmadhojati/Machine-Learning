#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

def SMOTE(X, y):
    """
    Synthetic Minority Oversampling Technique (SMOTE) for balancing class distribution.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target labels of shape (n_samples,).

    Returns:
    --------
    X_new : numpy.ndarray
        Feature matrix with synthetic samples added.
    y_new : numpy.ndarray
        Updated target labels with synthetic samples added.
    """

    # Finding the labels and the shape of the classes
    unique, counts = np.unique(y, return_counts=True)
    
    # Identify the minority class
    minority_class = np.where(counts == np.min(counts))
    
    # The number of Majority and Minority classes
    minority_count = dict(zip(unique, counts))[minority_class[0]]
    majority_count = dict(zip(unique, counts))[1 - minority_class[0]]
    
    # Separating the minority class for further analysis
    minority_data = X[y == minority_class[0]]
       
    # Ratio and remainder of the number of majority over minority classes 
    n = majority_count // minority_count
    m = majority_count % minority_count
    
    # Zeros matrix for the results
    X_n = np.zeros(((n - 1) * minority_count, minority_data.shape[1]))
    
    for i in range(n - 1):
        X_n[i * minority_count:(i + 1) * minority_count, :] = (minority_data + nearest_neighbour(minority_data).reshape(len(minority_data), 1))
    
    # If we have remainder
    if m != 0:
        rows = random.sample(range(0, len(minority_data)), m)
        X_m = minority_data[rows, :] + nearest_neighbour(minority_data[rows, :]).reshape(len(minority_data[rows, :]), 1)
        X_n = np.concatenate((X_n, X_m), axis=0)
        
    # Concatenate the produced observations to the original observations    
    X_new = np.concatenate((X, X_n), axis=0)
    
    # Setting class labels If 0 is the minority class
    if minority_class[0] == 0:
        y_new = np.concatenate((y, np.zeros(len(X_n))), axis=0)
    # If 1 is the minority class    
    else:
        y_new = np.concatenate((y, np.ones(len(X_n))), axis=0)
        
    return X_new, y_new  # The synthetic samples created by SMOTE

