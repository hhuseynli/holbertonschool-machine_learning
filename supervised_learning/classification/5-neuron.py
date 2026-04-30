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
        Propagates through one neuron of the neural network.

        Args:
            X - numpy.ndarray in the shape of (nx, m)

        Returns:
            __A - numpy.ndarray in the shape of (1, m)
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model
        Args:
        Y - shape (1,m) labels
        A - shape (1,m) activated output of each example

        Returns:
        cost - binary cross-entropy loss function (log loss)
        """

        losses = (1 - Y) * np.log(1.0000001 - A) + Y * np.log(A)
        cost = -losses.mean()

        return cost

    def evaluate(self, X, Y):
        """
        Args:
            X - numpy.ndarray in the shape (nx, m)
            Y - numpy.ndarray in the shape (1, m)

        Returns:
            prediction - neuron's prediction of X values - shape (1, m)
            cost - see cost function
        """
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Args:
            X -> numpy.ndarray of shape (nx, m)  = input data
            Y -> numpy.ndarray of shape (1, m) = correct output values
            A -> numpy.ndarray of shape (1, m) = activated output of each neuron

        Returns:
            nothing -> updates parameters W and b
        """
        m = X.shape[0]
        residuals = A-Y
        dJ_dw = np.matmul(X, residuals.T) / m # (nx, m) @ (m, 1) -> (nx, 1)
        dJ_db = np.mean(residuals)

        self.__W -= alpha * dJ_dw.T # (1, nx)
        self.__b -= alpha * dJ_db

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
