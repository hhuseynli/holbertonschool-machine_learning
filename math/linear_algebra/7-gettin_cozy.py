#!/usr/bin/env python3
""" Concat two matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concat based on an axis """
    cols = len(mat1[0])
    rows = len(mat1)
    if (rows, cols) != (len(mat2[0]), len(mat2)):
        return None
    if axis == 0:
        return mat1 + mat2
    elif axis == 1:
        return [mat1[i] + mat2[i] for i in range(cols)]
    return None
