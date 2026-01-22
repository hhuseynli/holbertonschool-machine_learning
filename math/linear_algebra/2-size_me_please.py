#!/usr/bin/env python3
""" Finds the shape of a matrix """


def matrix_shape(matrix):
    """ This function goes deeper in loops and creates a shape list """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
