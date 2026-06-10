#!/usr/bin/env python3

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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Initialize output layer
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
