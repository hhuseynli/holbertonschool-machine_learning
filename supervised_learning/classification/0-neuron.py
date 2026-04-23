#!/usr/bin/env python3
""" 0. Neuron """
import numpy as np


class Neuron:
    def __init__(self, nx):
        """ Validate input and set public attributes. """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(nx).reshape(nx, 1)
        self.b = 0
        self.A = 0
