#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import DBSCAN
from sklearn import metrics

def dbscan(X, eps, min_samples, labels_true):
    """
    Perform DBSCAN clustering and evaluate its performance using various metrics.

    Parameters:
    - X: Feature matrix.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    - labels_true: True cluster labels for evaluation.

    Prints:
    - Homogeneity, Completeness, V-measure, Adjusted Rand Index, and Adjusted Mutual Information scores.
    """

    # Compute DBSCAN
    db = DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples).fit(X)
    labels = db.labels_

    # Homogeneity
    hmgnty = metrics.homogeneity_score(labels_true, labels)
    print('Homogeneity =', hmgnty)

    # Completeness
    cmpltnss = metrics.completeness_score(labels_true, labels)
    print('Completeness =', cmpltnss)

    # V-measure
    vmeasr = metrics.v_measure_score(labels_true, labels)
    print('V_measure =', vmeasr)

    # Adjusted Rand Index
    adjrnd = metrics.adjusted_rand_score(labels_true, labels)
    print('Adjusted rand =', adjrnd)

    # Adjusted Mutual Information
    mutinfo = metrics.adjusted_mutual_info_score(labels_true, labels)
    print('Adjusted Mutual Information =', mutinfo)

