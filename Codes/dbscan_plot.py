#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

def dbscan_plot(X, eps, min_samples):
    """
    Plot the clusters formed by DBSCAN.

    Parameters:
    - X (array-like): Input features.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - None: Plots the clusters formed by DBSCAN.
    """

    # DBscan fit
    db = DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples).fit(X)
    labels = db.labels_

    # mask
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # unique labels
    unique_labels = set(labels)

    # number of clusters
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Color for each cluster
    colors = [plt.cm.autumn(l) for l in np.linspace(0, 1, len(unique_labels))]

    fig = plt.figure(figsize=(15, 10))

    # Plot
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black used for noise.
            color = [0, 0, 0, 1]

        cluster_points = (labels == label)

        xy = X[cluster_points & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markersize=15, markeredgecolor='k')

        xy = X[cluster_points & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markersize=7, markeredgecolor='k')

    plt.title('Number of clusters: %d' % n_clusters_)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

