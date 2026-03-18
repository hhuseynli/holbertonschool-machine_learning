#!/usr/bin/env python3
""" Calculate specificity """
import numpy as np


def specificity(confusion):
    """ in: confusion matrix, out: specificity for each class """
    total = confusion.sum()
    tp = np.diag(confusion)
    fn = confusion.sum(axis=1) - tp
    fp = confusion.sum(axis=0) - tp
    tn = total - (tp + fn + fp)
    return tn / (tn + fp)
