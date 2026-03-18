#!/usr/bin/env python3
""" Calculate sensitivity in a confusion matrix """
import numpy as np


def sensitivity(confusion);
    """ in: confusion matrix, out: sensitivity calculations"""
    # sensitivity of a class = value at class's number's index / overall sum
    sens = np.diagonal(confusion) / np.sum(confusion, axis=1)
    return sens
