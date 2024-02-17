#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import NearestNeighbors
import numpy as np

def nearest_neighbour(X):
    """
    Compute additive weights based on the distances to the nearest neighbors.

    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns:
    --------
    additive : numpy.ndarray
        Additive weights computed based on distances to nearest neighbors.
    """

    # Fit Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

    # Find distances and indices of nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    # Generate random weights between 0.001 and 1.0
    weight = np.random.uniform(low=0.001, high=1.0, size=X.shape[0])

    # Compute additive weights based on distances to the second nearest neighbor
    additive = distances[:, 1] * weight

    return additive

