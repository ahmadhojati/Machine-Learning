#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

def split(features, min_rows, max_depth, tresh, depth, result):
    """
    Recursively split the dataset to construct a decision tree.

    Parameters:
    - features: DataFrame containing the dataset.
    - min_rows: Minimum number of observations required in each branch for splitting.
    - max_depth: Maximum depth of the decision tree.
    - tresh: Threshold for impurity weight to terminate the tree.
    - depth: Depth of the current branch.
    - result: List to store the results of the split.

    Returns:
    - result: Updated list containing the results of the split.
    """

    # Get the column number, split criterion, and impurity weight using information gain
    cl_num, splt_crt, weight_Imp = Info_gain(features)

    # Split the dataset into left and right branches
    left, right = left_right(features, cl_num, splt_crt)

    # Terminate branches under certain conditions
    if depth >= max_depth:
        result.append([-1, -1, depth, 'left', np.array(terminate(left))[0]])
        result.append([-1, -1, depth, 'right', np.array(terminate(right))[0]])
        return result

    if len(left) <= min_rows or weight_Imp <= tresh:
        result.append([-1, -1, depth, 'left', np.array(terminate(left))[0]])
    else:
        result.append([Info_gain(left)[0], Info_gain(left)[1], depth, 'left'])
        split(left, min_rows, max_depth, tresh, depth + 1, result)

    if len(right) <= min_rows or weight_Imp <= tresh:
        result.append([-1, -1, depth, 'right', np.array(terminate(right))[0]])
    else:
        result.append([Info_gain(right)[0], Info_gain(right)[1], depth, 'right'])
        split(right, min_rows, max_depth, tresh, depth + 1, result)

    return result

# Example usage:
# result_tree = split(features, min_rows, max_depth, tresh, depth, result)

