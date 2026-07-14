#!/usr/bin/env python3
""" Shuffle data """

import numpy as np


def shuffle_data(X, Y):
    """ Shuffle """
    m = X.shape[0]
    # returns a list of shuffled indexes
    perm = np.random.permutation(m)
    return X[perm], Y[perm]
