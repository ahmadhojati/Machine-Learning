#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples

def KMeans_silhouette(X, n_clusters):
    """
    Compute and plot silhouette scores for K-Means clustering with different numbers of clusters.

    Parameters:
    - X (array-like): Input features.
    - n_clusters (int): The maximum number of clusters to try.

    Returns:
    - None: Plots silhouette scores for different numbers of clusters.
    """

    for i in range(2, n_clusters + 1):
        # Fit KMeans with the current number of clusters
        labels = KMeans(n_clusters=i).fit_predict(X)
        silhouette_values = silhouette_samples(X, labels)
        plot_silhouette(X, labels, silhouette_values)

def GMM_silhouette(X, n_clusters):
    """
    Compute and plot silhouette scores for Gaussian Mixture Model clustering with different numbers of components.

    Parameters:
    - X (array-like): Input features.
    - n_clusters (int): The maximum number of components to try.

    Returns:
    - None: Plots silhouette scores for different numbers of components.
    """

    for i in range(2, n_clusters + 1):
        # Fit Gaussian Mixture Model with the current number of components
        labels = GaussianMixture(n_components=i).fit_predict(X)
        silhouette_values = silhouette_samples(X, labels)
        plot_silhouette(X, labels, silhouette_values)

def plot_silhouette(X, labels, silhouette_values):
    """
    Plot the silhouette scores for each sample and the cluster plot.

    Parameters:
    - X (array-like): Input features.
    - labels (array-like): Cluster labels for each sample.
    - silhouette_values (array-like): Silhouette scores for each sample.

    Returns:
    - None: Plots silhouette scores and cluster plot.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 5])
    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_values = silhouette_values[labels == cluster]
        cluster_silhouette_values.sort()
        y_upper += len(cluster_silhouette_values)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_values, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_values)

    avg_score = np.mean(silhouette_values)
    ax1.axvline(avg_score, linestyle='--', linewidth=4, color='red')
    ax1.set_title('Silhouette Plot', y=1.02, fontsize=20)

    ax2.scatter(X[:, 0], X[:, 1], c=labels)
    ax2.set_title('Cluster Plot', y=1.02, fontsize=20)
    plt.show()

