#!/usr/bin/env python3
""" Neural network"""
import numpy as np


class NeuralNetwork:
    """ A neural network with one input, one hidden, and one output layer. """
    def __init__(self, nx, nodes):
        """ Args:
                nx => number of input features
                nodes => number of nodes in the hidden layer of the NN
        """

        # Input Validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialize output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """
        Args:
            X -> ndarray (nx, m), where nx = features; m = examples

        Returns: __A1 and __A2 -- activations of hidden and output layers
        """

        # (nodes, nx) @ (nx, m) = (nodes, m)
        # (nodes, m) + (1, m)[broadcasted] => (nodes, m)
        z1 = self.__W1 @ X + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        # (1, nodes) @ (nodes, m) = (1, m)
        z2 = self.__W2 @ self.__A1 + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
