#!/usr/bin/env python3
"""
Deep Neural Network for Multiclass Classification.

Provides a robust implementation of an L-layer Deep Neural Network with
configurable hidden layer activations ('sig' or 'tanh') and a stable
Softmax output layer for multi-categorical processing.
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """
        Initializes the deep neural network parameters and weights.

        Args:
            nx (int): Number of input features.
            layers (list):
            Number of nodes in each layer of the network.
            activation (str):
            Activation function for hidden layers ('sig' or 'tanh').

        Raises:
            TypeError:
            If nx is not an int or layers is not a valid list of ints.

            ValueError:
            If nx < 1, layers contains values <= 0, or activation is invalid.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__activation = activation
        self.__cache = {}
        self.__weights = {}

        for layer in range(self.__L):
            if not isinstance(layers[layer], int) or layers[layer] <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev = nx if layer == 0 else layers[layer - 1]

            # He (et al.) weight initialization
            W = np.random.randn(layers[layer], prev) * np.sqrt(2 / prev)
            b = np.zeros((layers[layer], 1))

            self.__weights[f"W{layer + 1}"] = W
            self.__weights[f"b{layer + 1}"] = b

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (A_last, cache)
                - A_last (numpy.ndarray): Activation output of the last layer.
                - cache (dict): Dictionary tracking all node activations.
        """
        prev = X
        self.__cache["A0"] = X

        for layer in range(self.__L):
            W = self.__weights[f"W{layer + 1}"]
            b = self.__weights[f"b{layer + 1}"]
            z = np.dot(W, prev) + b

            # Output layer utilizes Softmax for multiclass output
            if layer == self.__L - 1:
                prev = self.softmax(z)
            else:
                if self.__activation == 'sig':
                    prev = self.sigmoid(z)
                elif self.__activation == 'tanh':
                    prev = np.tanh(z)

            self.__cache[f"A{layer + 1}"] = prev

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the categorical cross-entropy loss.

        Args:
            Y (numpy.ndarray):
            One-hot encoded actual labels of shape (classes, m).

            A (numpy.ndarray):
            Output activations of the final layer of shape (classes, m).

        Returns:
            float: Categorical cross-entropy cost.
        """
        m = Y.shape[1]
        # Clip activations to avoid numerical underflow/log(0) issues
        A_clipped = np.clip(A, 1e-15, 1 - 1e-15)
        cost = -1 / m * np.sum(Y * np.log(A_clipped))
        return float(cost)

    def evaluate(self, X, Y):
        """
        Evaluates the network's predictions against target labels.

        Args:
            X (numpy.ndarray):
            Input data of shape (nx, m).

            Y (numpy.ndarray):
            One-hot encoded labels of shape (classes, m).

        Returns:
            tuple: (predictions, cost)
                - predictions (numpy.ndarray):
                One-hot array of predicted labels.
                - cost (float): Calculated loss value.
        """
        A, _ = self.forward_prop(X)
        m = X.shape[1]

        max_indices = np.argmax(A, axis=0)
        preds = np.zeros_like(A)
        preds[max_indices, np.arange(m)] = 1

        return preds, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one comprehensive backpropagation
        pass and updates parameters.
        Args:
            Y (numpy.ndarray):
            One-hot encoded actual labels of shape (classes, m).

            cache (dict):
            Dictionary tracking forward pass activation states.

            alpha (float):
            Learning rate parameter magnitude.
        """
        m = Y.shape[1]
        A_last = cache[f"A{self.__L}"]

        # Base error signal for Cross-Entropy combined with Softmax
        dZ = A_last - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache[f"A{layer - 1}"]
            Wl = self.__weights[f"W{layer}"]

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if layer > 1:
                A_current_layer = cache[f"A{layer - 1}"]
                if self.__activation == "sig":
                    dz_activation = A_current_layer * (1 - A_current_layer)
                elif self.__activation == "tanh":
                    dz_activation = 1 - A_current_layer ** 2

                dZ = np.dot(Wl.T, dZ) * dz_activation

            # Update structural parameter weights
            self.__weights[f"W{layer}"] -= alpha * dW
            self.__weights[f"b{layer}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network over a specified generation limit.

        Args:
            X (numpy.ndarray): Input training data of shape (nx, m).
            Y (numpy.ndarray): One-hot encoded training labels.
            iterations (int): Loop cycles to execute over.
            alpha (float): Hyperparameter learning rate scalar.
            verbose (bool): Toggles raw text matrix logs.
            graph (bool): Toggles final optimization charting.
            step (int): Training steps between performance snapshots.

        Returns:
            tuple: (predictions, cost)
            evaluated after final training execution.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps_list = []
        costs_list = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            if i % step == 0:
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    steps_list.append(i)
                    costs_list.append(cost)

            self.gradient_descent(Y, cache, alpha)

        final_A, final_cost = self.evaluate(X, Y)

        if verbose:
            print(f"Cost after {iterations} iterations: {final_cost}")

        if graph:
            steps_list.append(iterations)
            costs_list.append(final_cost)
            plt.plot(steps_list, costs_list, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return final_A, final_cost

    def save(self, filename):
        """Saves the instance state configuration into an external file."""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads and returns an instantiated network state object."""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def L(self):
        """int: The number of total layers in the neural network."""
        return self.__L

    @property
    def activation(self):
        """str: The hidden layer activation type string identifier."""
        return self.__activation

    @property
    def cache(self):
        """dict: Operational evaluation caching
        map capturing previous activations."""
        return self.__cache

    @property
    def weights(self):
        """dict: Mapping tracking parameter training arrays."""
        return self.__weights

    @staticmethod
    def sigmoid(x):
        """Computes basic scalar/elemental activation response equations."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(z):
        """Computes a stable column-wise multiclass probability matrix."""
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)
