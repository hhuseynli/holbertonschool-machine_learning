#!/usr/bin/env python3
""" This module will be calculating the adjugate """
cofactor = __import__("2-cofactor").cofactor


def adjugate(matrix):
    """ Adjugate is the transpose of the cofactor """
    return [list(col) for col in zip(*cofactor(matrix))]
