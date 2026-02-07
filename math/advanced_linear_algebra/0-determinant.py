#!/usr/bin/env python3
""" Calculate the determinant of a matrix """


def determinant(matrix):
    """ Validate and calculate """
    row_n = len(matrix)

    # Base case (before checking size to avoid errors)
    if matrix == [[]]:
        return 1

    valid_rows = sum([isinstance(row, list) for row in matrix])

    # Check type
    if not isinstance(matrix, list) or row_n == 0 or \
            valid_rows != row_n:
        raise TypeError("matrix must be a list of lists")

    # Check for jagged matrices
    first_row = len(matrix[0])
    eq_rows = sum([len(row) == first_row for row in matrix])
    col_n = first_row if eq_rows == row_n else 0

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
