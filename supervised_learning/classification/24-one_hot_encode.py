#!/usr/bin/env python3
"""One-hot encoding"""
import numpy as np


def one_hot_encode(Y, classes):
    """ encode Y """
    if not (isinstance(Y, np.ndarray)):
        return None
    if not (isinstance(classes, int)):
        return None
    if classes < 2:
        return None
    if classes < np.max(Y):
        return None
    res = np.zeros((len(Y), classes))
    for i in range(len(Y)):
        temp = np.zeros((1, classes))
        temp[0, Y[i]] = 1
        res[i] = temp
    return res.T
