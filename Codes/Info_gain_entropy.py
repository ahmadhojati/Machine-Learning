#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

def Info_gain_entropy(data):
    """
    Calculate information gain and entropy for each feature using the entropy criterion.

    Parameters:
    - data: DataFrame containing the dataset.

    Returns:
    - col_min: Column number of the feature with the minimum entropy.
    - splt_crt: Split criterion for the selected feature.
    """

    entropy = np.zeros((len(data), len(data.columns) - 1))

    for i in range(len(data.columns) - 1):
        for j in range(len(data)):
            splt_crt = data.iloc[j, i]
            left, right = left_right(data, i, splt_crt)
            Y_l = left.iloc[:, -1]
            Y_r = right.iloc[:, -1]

            # Calculate left impurity (entropy)
            if len(Y_l) == 0:
                left_Imp = 0
            else:
                p_l = sum(Y_l) / len(Y_l)
                if p_l == 0 or p_l == 1:
                    a = 0
                    b = 0
                else:
                    a = np.log2(p_l)
                    b = np.log2(1 - p_l)
                left_Imp = -float(p_l) * float(a) - (1 - float(p_l)) * float(b)

            # Calculate right impurity (entropy)
            if len(Y_r) == 0:
                right_Imp = 0
            else:
                p_r = sum(Y_r) / len(Y_r)
                if p_r == 0 or p_r == 1:
                    a = 0
                    b = 0
                else:
                    a = np.log2(p_r)
                    b = np.log2(1 - p_r)
                right_Imp = -p_r * a - (1 - p_r) * b

            # Calculate entropy
            entropy[j, i] = (len(Y_l) / len(data)) * left_Imp + (len(Y_r) / len(data)) * right_Imp

    # Find the row and column of the minimum entropy
    row_min = np.argmin(np.min(entropy, axis=1))
    col_min = np.argmin(np.min(entropy, axis=0))
    splt_crt = data.iloc[row_min, col_min]

    return col_min, splt_crt

# Example usage:
# col_min_entropy, splt_crt_entropy = Info_gain_entropy(data)

