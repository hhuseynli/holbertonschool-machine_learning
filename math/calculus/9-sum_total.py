#!/usr/bin/env python3
""" Calculates a sum """


def summation_i_squared(n):
    """ let's use computational power to do this than formulas """
    """ because we're cool """
    sum = 0
    for i in range(1, n+1):
        sum += i ** 2
    return sum
