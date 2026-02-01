#!/usr/bin/env python3
""" Slice with python slice() function and numpy """


def np_slice(matrix, axes={}):
    """ Slice based on a dictionary """

    # Fill in the gaps of axes
    for dim, size in enumerate(matrix.shape):
        if dim not in axes:
            axes[dim] = (size, )

    # Convert to slices
    axes_slices = list(matrix.shape)
    for axis in axes:
        form = axes[axis]
        axes_slices[axis] = slice(*form)

    # Unpack slices to slice the matrix
    return matrix[*axes_slices]
