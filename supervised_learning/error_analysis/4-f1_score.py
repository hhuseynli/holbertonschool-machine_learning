#!/usr/bin/env python3
""" F1 score is the harmonic mean of sensitivity and precision """
precision = __import__('2-precision.py').precision
sensitivity = __import__('1-sensitivity.py').sensitivity


def f1_score(confusion):
    """ in: conf. matrix, out: f-score """
    n = confusion.shape[0]
    return n * (1 / precision(confusion) + 1 / sensitivity(confusion))
