#!/usr/bin/env python3
"""
Module containing a Neural Network with one hidden layer
and binary classification capabilities.
"""
import numpy as np


class NeuralNetwork:
    """
    A neural network with one input layer, one hidden layer,
    and one output layer performing binary classification.
    """
    def __init__(self, nx, nodes):
        """
        Initializes the neural network structure.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes are not integers.
            ValueError: If nx or nodes are less than 1.
        """
        # Input Validation for features
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Input Validation for hidden nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Weights matrix initialized with random normal distribution
        # Shape: (nodes, nx)
        self.__W1 = np.random.randn(nodes, nx)

        # Biases vector initialized as zeros
        # Shape: (nodes, 1)
        self.__b1 = np.zeros((nodes, 1))

        # Activated output (prediction) for hidden layer
        self.__A1 = 0

        # Weights matrix for output layer
        # Shape: (1, nodes)
        self.__W2 = np.random.randn(1, nodes)

        # Bias scalar/vector for output layer
        self.__b2 = 0

        # Activated output (prediction) for output layer
        self.__A2 = 0

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
                nx is the number of input features.
                m is the number of examples.

        Returns:
            tuple: (A1, A2) where:
                A1 is the activated output of the hidden layer.
                A2 is the activated output of the output layer.
        """
        # Hidden Layer: (nodes, nx) @ (nx, m) -> (nodes, m)
        z1 = self.__W1 @ X + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        # Output Layer: (1, nodes) @ (nodes, m) -> (1, m)
        z2 = self.__W2 @ self.__A1 + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using binary cross-entropy.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output of the output layer (1, m).

        Returns:
            float: The calculated binary cross-entropy cost.
        """
        # Elements scaled safely away from 1.0 to avoid log of 0 errors
        losses = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -np.mean(losses)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's classifications and current loss.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            tuple: (pred, cost) where:
                pred is a numpy.ndarray (1, m) of 0s and 1s representing
                     predictions.
                cost is the binary cross-entropy loss.
        """
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Vectorized classification split without using native Python loops
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A1 (numpy.ndarray): Activations of the hidden layer (nodes, m).
            A2 (numpy.ndarray): Activations of the output layer (1, m).
            alpha (float): The learning rate parameter.
        """
        m = X.shape[1]

        # 1. Output Layer Gradients
        dz2 = A2 - Y
        dw2 = (1 / m) * (dz2 @ A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # 2. Hidden Layer Gradients using element-wise derivative multiplication
        dz1 = (self.__W2.T @ dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * (dz1 @ X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # 3. Parameter Update Phase
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network over a set number of optimization steps.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Total training steps to run.
            alpha (float): Learning rate threshold.

        Raises:
            TypeError: If iterations is not an int or alpha is not a float.
            ValueError: If iterations or alpha are less than or equal to 0.

        Returns:
            tuple: Evaluation outcomes (predictions, cost) after training.
        """
        # Exception tracking for iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Exception tracking for alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Optimization iteration loop
        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)

    @property
    def W1(self):
        """
        Gets the private weights matrix of the hidden layer.

        Returns:
            numpy.ndarray: The weights matrix W1.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Gets the private bias vector of the hidden layer.

        Returns:
            numpy.ndarray: The bias vector b1.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Gets the private activated output matrix of the hidden layer.

        Returns:
            numpy.ndarray: The activation matrix A1.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Gets the private weights matrix of the output layer.

        Returns:
            numpy.ndarray: The weights matrix W2.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Gets the private bias parameter of the output layer.

        Returns:
            int or float or numpy.ndarray: The bias parameter b2.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Gets the private prediction output matrix of the output layer.

        Returns:
            numpy.ndarray: The activation matrix A2.
        """
        return self.__A2