#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import metrics

def acc_KM(labels_true, labels):
    """
    Evaluate the performance of a clustering algorithm using various clustering metrics.

    Parameters:
    - labels_true: True cluster labels.
    - labels: Predicted cluster labels.

    Prints:
    - Homogeneity, Completeness, V-measure, Adjusted Rand Index, and Adjusted Mutual Information scores.
    """

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

# Example usage:
# acc_KM(labels_true, labels)

