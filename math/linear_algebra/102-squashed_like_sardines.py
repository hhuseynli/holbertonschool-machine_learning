#!/usr/bin/env python3
""" Concatenate matrices along a specific axes """


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis."""

    def get_shape(mat):
        """Get the shape of a matrix as a tuple."""
        shape = []
        current = mat
        while isinstance(current, list):
            shape.append(len(current))
            if len(current) == 0:
                break
            current = current[0]
        return tuple(shape)

    def deep_copy(mat):
        """Create a deep copy of a matrix."""
        if not isinstance(mat, list):
            return mat
        return [deep_copy(item) for item in mat]

    # Get shapes of both matrices
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Check if they have the same number of dimensions
    if len(shape1) != len(shape2):
        return None

    # Check if axis is valid
    if axis < 0 or axis >= len(shape1):
        return None

    # Check if all dimensions except the axis dimension match
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    # Perform concatenation
    if axis == 0:
        # Concatenate at the outermost level
        return deep_copy(mat1) + deep_copy(mat2)
    else:
        # Need to recursively concatenate at deeper levels
        result = []
        for i in range(len(mat1)):
            concatenated = cat_matrices(mat1[i], mat2[i], axis=axis-1)
            if concatenated is None:
                return None
            result.append(concatenated)
        return result
