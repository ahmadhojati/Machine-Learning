#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def split_entropy(features, min_rows, max_depth, depth, result):
    """
    Recursively split the dataset using the entropy criterion to construct a decision tree.

    Parameters:
    - features: DataFrame containing the dataset.
    - min_rows: Minimum number of observations required in each branch for splitting.
    - max_depth: Maximum depth of the decision tree.
    - depth: Depth of the current branch.
    - result: List to store the results of the split.

    Returns:
    - result: Updated list containing the results of the split.
    """

    # Get the column number and split criterion using entropy criterion
    cl_num, splt_crt = Info_gain_entropy(features)

    # Split the dataset into left and right branches
    left, right = left_right(features, cl_num, splt_crt)

    # Terminate branches under certain conditions
    if depth >= max_depth:
        result.append([-1, -1, depth, 'left', np.array(terminate(left))[0]])
        result.append([-1, -1, depth, 'right', np.array(terminate(right))[0]])
        return result

    if len(left) <= min_rows:
        result.append([-1, -1, depth, 'left', np.array(terminate(left))[0]])
    else:
        result.append([Info_gain_entropy(left)[0], Info_gain_entropy(left)[1], depth, 'left'])
        split_entropy(left, min_rows, max_depth, depth + 1, result)

    if len(right) <= min_rows:
        result.append([-1, -1, depth, 'right', np.array(terminate(right))[0]])
    else:
        result.append([Info_gain_entropy(right)[0], Info_gain_entropy(right)[1], depth, 'right'])
        split_entropy(right, min_rows, max_depth, depth + 1, result)

    return result

# Example usage:
# result_tree_entropy = split_entropy(features, min_rows, max_depth, depth, result)

