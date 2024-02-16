#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN

def evaluate_clustering(X, labels_true, model, **kwargs):
    """
    Evaluate the clustering performance using various metrics.

    Parameters:
    - X (array-like): Input features.
    - labels_true (array-like): True labels of the data.
    - model (str): Clustering algorithm to evaluate ('kmeans' or 'dbscan').
    - **kwargs: Additional parameters for the clustering model.

    Returns:
    - dict: Dictionary containing different clustering performance metrics.
    """

    if model == 'kmeans':
        # KMeans clustering
        kmeans = KMeans(**kwargs)
        labels = kmeans.fit_predict(X)
    elif model == 'dbscan':
        # DBSCAN clustering
        dbscan = DBSCAN(**kwargs)
        labels = dbscan.fit_predict(X)
    else:
        raise ValueError("Invalid clustering model. Use 'kmeans' or 'dbscan'.")

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
    if model == 'kmeans':
        # Silhouette is not available for DBSCAN
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

    print('---------------------------------------------------')

    # Return metrics in a dictionary
    metrics_dict = {
        'homogeneity': hmgnty,
        'completeness': cmpltnss,
        'v_measure': vmeasr,
        'adjusted_rand': adjrnd,
        'adjusted_mutual_info': mutinfo
    }

    return metrics_dict

# Example usage for KMeans clustering
# Replace 'your_data' and 'your_true_labels' with the actual data and true labels
# kmeans_metrics = evaluate_clustering(your_data, your_true_labels, model='kmeans', n_clusters=3)
# print(kmeans_metrics)

# Example usage for DBSCAN clustering
# Replace 'your_data' and 'your_true_labels' with the actual data and true labels
# dbscan_metrics = evaluate_clustering(your_data, your_true_labels, model='dbscan', eps=0.5, min_samples=5)
# print(dbscan_metrics)

