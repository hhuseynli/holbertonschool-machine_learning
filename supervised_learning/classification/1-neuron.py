#!/usr/bin/env python3
""" 1. Neuron """
import numpy as np


class Neuron:
    """ A neuron is the simplest building block of a neural network."""
    def __init__(self, nx):
        """ Validate input and set private attributes of the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
