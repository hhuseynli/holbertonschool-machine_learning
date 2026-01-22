#!/usr/bin/env python3
""" Concat two matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concat based on an axis """
    if (len(mat1[0]) == len(mat2[0])) and (axis == 0):
        concat = [row[:] for row in (mat1 + mat2)]
        return concat
    elif (len(mat1) == len(mat2)) and (axis == 1):
        concat = [mat1[j] + mat2[j] for j in range(len(mat1))]
        return concat
    else:
        return None
