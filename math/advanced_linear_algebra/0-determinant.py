#!/usr/bin/env python3
""" Calculate the determinant of a matrix """


def determinant(matrix):
    """ Validate and calculate """
    # Check type
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check size before taking matrix[0]
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    row_n = len(matrix)
    col_n = len(matrix[0])

    # Base case (before checking size to avoid errors)
    if matrix == [[]]:
        return 1

    # Single-numbered base case
    if row_n == 1 and col_n == 1:
        return matrix[0][0]

    # Check size
    if row_n != col_n:
        raise ValueError("matrix must be a square matrix")

    col_sum = list()

    # go over each column (only the first element is necessary)
    for ind, num in enumerate(matrix[0]):

        # remove the first row and ind'th column to keep the minor
        minor = [[matrix[i][j] for j in range(col_n) if j != ind]
                 for i in range(row_n) if i != 0]

        # calculate sign based on ind (even ir odd)
        sign = 1 if ind % 2 == 0 else -1

        # add this to the list that will be summed up
        col_sum.append(sign * num * determinant(minor))

    return sum(col_sum)
