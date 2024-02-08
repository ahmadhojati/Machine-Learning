#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def dbscan_plot(X, eps, min_samples):
    """
    Perform DBSCAN clustering on 2D data and visualize the clusters.

    Parameters:
    - X: Feature matrix (2D data).
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Displays a plot showing the identified clusters.
    """

    # DBscan fit
    db = DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples).fit(X)
    labels = db.labels_

    # Mask for core points
    core_mask = np.zeros_like(db.labels_, dtype=bool)
    core_mask[db.core_sample_indices_] = True

    # Unique labels
    unique_labels = set(labels)

    # Number of clusters
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Color for each cluster
    colors = [plt.cm.autumn(l) for l in np.linspace(0, 1, len(unique_labels))]

    fig = plt.figure(figsize=(15, 10))

    # Plot
    for cluster_label, color in zip(unique_labels, colors):
        if cluster_label == -1:
            # Black used for noise.
            color = [0, 0, 0, 1]

        cluster_points = (labels == cluster_label)

        # Plot core points
        core_points = X[cluster_points & core_mask]
        plt.plot(core_points[:, 0], core_points[:, 1], 'o', markerfacecolor=tuple(color), markersize=15,
                 markeredgecolor='k')

        # Plot non-core points
        non_core_points = X[cluster_points & ~core_mask]
        plt.plot(non_core_points[:, 0], non_core_points[:, 1], 'o', markerfacecolor=tuple(color), markersize=7,
                 markeredgecolor='k')

    plt.title('Number of clusters: %d' % n_clusters_)
    plt.show()

# Example usage:
# dbscan_plot(X, eps=0.5, min_samples=5)

