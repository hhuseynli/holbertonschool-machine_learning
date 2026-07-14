#!/usr/bin/env python3
""" Mini batches """

import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """ Returns a list of mini-batches """
    m = X.shape[0]
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    batches = list()
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i: i+batch_size]
        Y_batch = Y_shuffled[i: i+batch_size]
        batches.append((X_batch, Y_batch))
    return batches
