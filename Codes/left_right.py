#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def left_right(data, cl_num, splt_crt):
    """
    Split the dataset into left and right subsets based on the split criterion.

    Parameters:
    - data: DataFrame containing the dataset.
    - cl_num: Column number of the data used for splitting.
    - splt_crt: Split criterion.

    Returns:
    - left: Subset of the dataset where the values in the specified column are less than the split criterion.
    - right: Subset of the dataset where the values in the specified column are greater than or equal to the split criterion.
    """

    # Extract target variable (Y) and features (X)
    Y = data.iloc[:, -1]
    X = data.iloc[:, 0:len(data)-1]

    # Split the dataset into left and right subsets based on the split criterion
    left = X[X.iloc[:, cl_num] < splt_crt]
    right = X[X.iloc[:, cl_num] >= splt_crt]

    return left, right

# Example usage:
# left_subset, right_subset = left_right(data, cl_num, splt_crt)

