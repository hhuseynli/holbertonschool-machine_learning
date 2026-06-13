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

    def cost(self, Y, A):
        """ Calculate the cost using binary cross-entropy loss
        Args:
            Y -> ndarray (1, m) with the correct labels
            A -> ndarray (1, m) with the activated output

        Returns:
            cost (BCE)
        """
        losses = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -np.mean(losses)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions """
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = (A >= 0.5).astype(int)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural networks
        Args:
        X -> ndarray (nx, m), where nx = features; m = examples
        Y -> ndarray (1, m) with the correct labels
        A1 -> ndarray (nodes, m) with the activations of the hidden layer
        A2 -> ndarray (1, m) with the activation of the output layer
        """
        m = X.shape[1]  # Number of training examples

        # 1. Output Layer Gradients
        dz2 = A2 - Y
        dw2 = (1 / m) * (dz2 @ A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # 2. Hidden Layer Gradients
        dz1 = (self.__W2.T @ dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * (dz1 @ X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # 3. Update Weights and Biases
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neural network by running forward propagation and
        gradient descent for a given number of iterations.
        Args:
            X => ndarray (nx, m) containing the input data
            Y => ndarray (1, m) containing the correct labels
            iterations => number of iterations to train over
            alpha => learning rate
        Returns:
            The evaluation of the training data after training completes.
        """
        # Exception handling for iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Exception handling for alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)

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
