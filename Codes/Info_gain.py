#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

def Info_gain(data):
    """
    Calculate information gain, Gini index, and impurity weight for each feature.

    Parameters:
    - data: DataFrame containing the dataset.

    Returns:
    - col_min: Column number of the feature with the minimum Gini index.
    - splt_crt: Split criterion for the selected feature.
    - weight_Imp: Impurity weight for the selected feature.
    """

    gini = np.zeros((len(data), len(data.columns) - 1))
    w_Imp = np.zeros((len(data), len(data.columns) - 1))

    for i in range(len(data.columns) - 1):
        for j in range(len(data)):
            splt_crt = data.iloc[j, i]
            left, right = left_right(data, i, splt_crt)
            Y_l = left.iloc[:, -1]
            Y_r = right.iloc[:, -1]

            # Calculate left impurity
            left_Imp = 0 if len(Y_l) == 0 else 1 - (sum(Y_l) / len(Y_l))**2 - ((len(Y_l) - sum(Y_l)) / len(Y_l))**2

            # Calculate right impurity
            right_Imp = 0 if len(Y_r) == 0 else 1 - (sum(Y_r) / len(Y_r))**2 - ((len(Y_r) - sum(Y_r)) / len(Y_r))**2

            # Calculate impurity
            Imp = 1 - (len(Y_l) / len(data))**2 - (len(Y_r) / len(data))**2

            # Calculate information gain
            gain = Imp - (len(Y_l) / len(data)) * left_Imp - (len(Y_r) / len(data)) * right_Imp

            # Calculate impurity weight
            w_Imp[j, i] = (len(Y_l) + len(Y_r)) / len(data) * gain

            # Calculate Gini index
            gini[j, i] = (len(Y_l) / len(data)) * left_Imp + (len(Y_r) / len(data)) * right_Imp

    # Find the row and column of the minimum Gini index
    row_min = np.argmin(np.min(gini, axis=1))
    col_min = np.argmin(np.min(gini, axis=0))

    # Get the split criterion and impurity weight for the selected feature
    splt_crt = data.iloc[row_min, col_min]
    weight_Imp = w_Imp[row_min, col_min]

    return col_min, splt_crt, weight_Imp

# Example usage:
# col_min, splt_crt, weight_Imp = Info_gain(data)

