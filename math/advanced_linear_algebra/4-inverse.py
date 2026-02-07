#!/usr/bin/env python3
""" Find the inverse of a matrix """
determinant = __import__("0-determinant").determinant
adjugate = __import__("3-adjugate").adjugate


def inverse(matrix):
    """ Calculate the inverse """
    det = determinant(matrix)
    adj = adjugate(matrix)
    if det == 0:
        return None

    row_n = len(matrix)
    col_n = len(matrix[0])

    return [[adj[i][j] / det for j in range(col_n)] for i in range(row_n)]
