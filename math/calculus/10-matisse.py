#!/usr/bin/env python3
""" Differentiate and get coefficients """


def poly_derivative(poly):
    """ Return derivative of a polynomial """
    if not isinstance(poly, list):
        return None
    diff = []
    total = 0
    for i in range(1, len(poly)):
        coefficient = i * poly[i]
        total += coefficient
        diff.append(coefficient)
    if total == 0:
        return [0]
    return diff
