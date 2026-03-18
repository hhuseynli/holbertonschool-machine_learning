#!/usr/bin/env python3
""" F1 score is the harmonic mean of sensitivity and precision """
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    """ in: conf. matrix, out: f-score """
    return 2 * (1 / precision(confusion) + 1 / sensitivity(confusion))
