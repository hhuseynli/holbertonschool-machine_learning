#!/usr/bin/env python3
""" Transpose of a matrix """


def matrix_transpose(matrix):
    """ A transpose is when each column is turned into a row """
    transpose = []
    cols = len(matrix[0])  # number of cols
    rows = len(matrix)  # number of rows
    for i in range(cols):  # for each col append a row
        col = [matrix[j][i] for j in range(rows)]  # get the i'th column
        transpose.append(col)  # append col as a row
    return transpose
