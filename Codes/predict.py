#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def predict(data, tree, row):
    """
    Predict the outcome for a given test data point using a decision tree.

    Parameters:
    - data: DataFrame containing the test data.
    - tree: DataFrame representing the decision tree in Materialized Path format.
    - row: Index of the test data point.

    Returns:
    - prediction: Predicted outcome (0 or 1).
    """

    depth = 0
    n = len(tree) - 1
    d = max(tree['Depth'])
    row_b = 0

    while tree['Index'][row_b] >= 0:
        if data.loc[row][int(tree['Index'][row_b])] < tree['Criteria'][row_b]:
            depth = depth + 1
            row_b = int(tree[(tree['Depth'] == depth) & (tree['branch'] == 'left')].index.tolist()[0])
            row_end = int(tree[(tree['Depth'] == depth) & (tree['branch'] == 'right')].index.tolist()[0])
            tree = tree.loc[row_b:row_end, :]
            prediction = tree['value'][row_b]

        else:
            depth = depth + 1
            row_b = int(tree[(tree['Depth'] == depth) & (tree['branch'] == 'right')].index.tolist()[0])
            tree = tree.loc[row_b:, :]
            prediction = tree['value'][row_b]

    return prediction

# Example usage:
# result_prediction = predict(test_data, decision_tree, row_index)

