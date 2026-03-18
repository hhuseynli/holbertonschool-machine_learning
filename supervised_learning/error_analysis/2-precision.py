#!/usr/bin/env python3
""" Calculate precision of a confusion matrix """
import numpy as np


def precision(confusion):
    """ in: confusion matrix, out: precision value of each row """
    tp = np.diagonal(confusion)
    pred_positives = np.sum(confusion, axis=0)
    return tp / pred_positives
