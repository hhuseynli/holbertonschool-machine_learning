#!/usr/bin/env python3
""" Moving average """

import numpy as np


def moving_average(data, beta):
    """ Moving average """
    moving_averages = []
    v = 0
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        moving_average = v / (1 - beta ** (i + 1))
        moving_averages.append(moving_average)

    return moving_averages
