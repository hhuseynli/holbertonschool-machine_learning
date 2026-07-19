#!/usr/bin/env python3
""" L2 Regularization """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Returns the cost of the network accounting for L2 regularization """
    for l in range(1, L+1):
        cost += lambtha * np.sum(weights[f"W{l}"]) ** 2 / (2 * m)
    return cost