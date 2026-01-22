#!/usr/bin/env python3
""" Add lists element-wise """


def add_arrays(arr1, arr2):
    """ Check for size and then add """
    size = len(arr1)
    if size != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(size)]
