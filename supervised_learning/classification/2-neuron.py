#!/usr/bin/env python3
""" 2. Neuron """
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

    def forward_prop(self, X):
        """
        Propagates through one layer of the neural network.

        Args:
            X - numpy.ndarray in the shape of (nx, m)

        Returns:
            __A -
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
