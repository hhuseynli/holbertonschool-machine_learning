#!/usr/bin/env python3
""" Matrix multiplication """


def mat_mul(mat1, mat2):
    """ Get dot product with transverse """
    if (len(mat1[0]) != len(mat2)):
        return None
    return [
        [
            sum(m[i] * n[i] for i in range(len(mat2)))
            for n in zip(*mat2)
        ]
        for m in mat1
    ]
