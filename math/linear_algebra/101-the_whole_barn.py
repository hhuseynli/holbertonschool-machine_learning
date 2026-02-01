#!/usr/bin/env python3
""" Matrix addition"""


def add_matrices(mat1, mat2):
    """Utilize a recursive solution"""

    # Base case: both are numbers (not lists)
    if not isinstance(mat1, list) and not isinstance(mat2, list):
        return mat1 + mat2

    # If one is a list and the other isn't, shapes don't match
    if isinstance(mat1, list) != isinstance(mat2, list):
        return None

    # Both are lists - check if they have the same length
    if len(mat1) != len(mat2):
        return None

    # Recursively add corresponding elements
    result = []
    for i in range(len(mat1)):
        added = add_matrices(mat1[i], mat2[i])
        if added is None:
            return None
        result.append(added)

    return result
