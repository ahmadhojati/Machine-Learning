#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def hierarchical(df, n_cls):
    """
    Perform hierarchical clustering on a DataFrame.

    Parameters:
    - df: Input DataFrame containing data points.
    - n_cls: The desired number of clusters.

    Returns:
    - cls: List of clusters after hierarchical clustering using complete linkage.
    """

    # Extract the values from the DataFrame
    d = df.values

    # Initialize clusters with each row as a separate cluster
    init = [[row.tolist()] for row in d]

    # Perform hierarchical clustering using complete linkage
    return complete_linkage(init, n_cls)

# Example usage:
# result_clusters = hierarchical(df, n_cls)

