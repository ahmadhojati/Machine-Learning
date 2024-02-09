#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.spatial import distance

def calculate_farthest_distance(cluster1, cluster2):
    """
    Calculate the farthest distance between points in two clusters.

    Parameters:
    - cluster1: Points in the first cluster.
    - cluster2: Points in the second cluster.

    Returns:
    - farthest_distance: The farthest distance between any pair of points in the two clusters.
    """
    farthest_distance = -np.inf

    # Iterate over each point in each cluster
    for point1 in cluster1:
        for point2 in cluster2:
            # Update the farthest distance if a greater distance is found
            current_distance = distance.euclidean(point1, point2)
            if farthest_distance < current_distance:
                farthest_distance = current_distance

    return farthest_distance

