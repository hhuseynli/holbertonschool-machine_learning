#!/usr/bin/env python3
""" F1 score is the harmonic mean of sensitivity and precision """
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    """ in: conf. matrix, out: f-score """
    prec = precision(confusion)
    sens = sensitivity(confusion)

    return 2 * (prec * sens) / (prec + sens)
