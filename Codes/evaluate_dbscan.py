#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import DBSCAN
from sklearn import metrics

def evaluate_dbscan(X, eps, min_samples, labels_true):
    """
    Evaluate DBSCAN clustering performance using various metrics.

    Parameters:
    - X (array-like): Input features.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    - labels_true (array-like): True labels of the data.

    Returns:
    - None: Prints the clustering performance metrics.
    """

    # Compute DBSCAN
    db = DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples).fit(X)
    labels = db.labels_

    # Homogeneity
    hmgnty = metrics.homogeneity_score(labels_true, labels)
    print('Homogeneity = ', hmgnty)

    # Completeness
    cmpltnss = metrics.completeness_score(labels_true, labels)
    print('Completeness = ', cmpltnss)

    # V-measure
    vmeasr = metrics.v_measure_score(labels_true, labels)
    print('V_measure = ', vmeasr)

    # Adjusted Rand Index
    adjrnd = metrics.adjusted_rand_score(labels_true, labels)
    print('Adjusted rand = ', adjrnd)

    # Adjusted Mutual Information
    mutinfo = metrics.adjusted_mutual_info_score(labels_true, labels)
    print('Adjusted Mutual Information', mutinfo)

    # Silhouette
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

