#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.linear_model import LogisticRegression

def AdaBoost(X, y, n_estimator):
    """
    Implement the AdaBoost algorithm using logistic regression as the base classifier.

    Parameters:
    - X: Feature matrix.
    - y: Target labels.
    - n_estimator: Number of weak classifiers (base learners) to be trained.

    Returns:
    - clf_list: List of trained classifiers.
    - amount_of_say_list: List of the amount of say (importance) for each classifier.
    """

    # Initial variables
    n = len(y)
    clf_list = []
    amount_of_say_list = []
    smpl_weight_list = []

    # Initial sample weights
    smpl_weight = np.ones(n) / n
    smpl_weight_list.append(smpl_weight.copy())

    for m in range(n_estimator):
        # Fit a logistic regression classifier
        clf = LogisticRegression()
        clf.fit(X, y, sample_weight=smpl_weight)
        y_p = clf.predict(X)

        # Misclassifications
        incorrect = (y_p != y)

        # Total error
        Total_error = np.mean(np.average(incorrect, weights=smpl_weight, axis=0))

        # Amount of Say
        amount_of_say = 0.5 * np.log((1 - Total_error) / Total_error)

        # Sample weights update using AdaBoost weight update formula
        smpl_weight *= np.exp(amount_of_say * incorrect * ((smpl_weight > 0) | (amount_of_say < 0)))

        # Append the computed values
        clf_list.append(clf)
        amount_of_say_list.append(amount_of_say.copy())

    # Convert lists to numpy arrays
    clf_list = np.asarray(clf_list)
    amount_of_say_list = np.asarray(amount_of_say_list)

    return clf_list, amount_of_say_list

