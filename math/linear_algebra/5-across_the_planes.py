#!/usr/bin/env python3
""" Element-wise 2D addition of matrices """


def add_matrices2D(mat1, mat2):
    """ Check for shape and then add them """
    shape1 = (len(mat1), len(mat1[0]))
    shape2 = (len(mat2), len(mat2[0]))
    if shape1 != shape2:
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(shape1[1])] for i in range(shape1[0])]
