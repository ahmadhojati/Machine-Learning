#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def complete_linkage(cls, n_cls):
    """
    Perform hierarchical clustering using complete linkage.

    Parameters:
    - cls: List of clusters, each represented as a list of points.
    - n_cls: The desired number of clusters.

    Returns:
    - cls: List of clusters after hierarchical clustering using complete linkage.
    """

    while len(cls) - n_cls != 0:
        # Clustering
        close_dist = np.inf
        clust_1 = clust_2 = None

        # Iterate over every cluster (until the second last element)
        for id1, cls1 in enumerate(cls[:len(cls) - 1]):
            for id2, cls2 in enumerate(cls[(id1 + 1):]):
                far_dist = dist(cls1, cls2)

                if far_dist < close_dist:
                    clust_1 = id1
                    clust_2 = id1 + id2 + 1
                    close_dist = far_dist

        # Extend: appends the contents of the second cluster to the first without flattening it out
        cls[clust_1].extend(cls[clust_2])

        # Remove the second cluster as it has been merged into the first
        cls.pop(clust_2)

    return cls

