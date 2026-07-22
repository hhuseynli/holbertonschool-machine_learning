#!/usr/bin/env python3
""" L2 Regularization """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Returns the cost of the network accounting for L2 regularization """
    frobenius = 0
    for layer in range(1, L+1):
        frobenius += np.sum(weights[f"W{layer}"] ** 2)
    reg_term = lambtha * frobenius / (2 * m)
    return cost + reg_term
