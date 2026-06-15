#!/usr/bin/env python3
"""Deep Neural Network"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """ A deep neural network consists of multiple hidden layers """
    def __init__(self, nx, layers, activation='sig'):
        if not (isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not (isinstance(layers, list)) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        for layer in range(self.__L):
            if layers[layer] <= 0 or not (isinstance(layers[layer], int)):
                raise TypeError("layers must be a list of positive integers")
            if layer == 0:
                prev = nx
            else:
                prev = layers[layer-1]
            W = (np.random.randn(layers[layer], prev) *
                 np.sqrt(2 / prev))
            b = np.zeros((layers[layer], 1))
            self.__weights.update({f"W{layer+1}": W})
            self.__weights.update({f"b{layer+1}": b})

    def forward_prop(self, X):
        """ Calculates the activation of each neuron
        Args:
            X -> ndarray (nx, m), where nx = features; m = examples
        Returns:
            Y, cache
        """
        prev = X
        self.__cache["A0"] = X
        for layer in range(self.__L):
            z = self.__weights[f"W{layer+1}"] @ prev \
                + self.__weights[f"b{layer+1}"]
            if layer == self.__L - 1:
                prev = self.softmax(z)
            else:
                if self.__activation == "sig":
                    prev = self.sigmoid(z)
                elif self.__activation == "tanh":
                    prev = np.tanh(z)
            self.__cache[f"A{layer+1}"] = prev
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost using multiclass cross entropy loss """
        # To avoid division by zero or log(0), clip or add an offset.
        # Categorical cross-entropy formula: -1/m * sum(Y * log(A))
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluate the neural network's predictions """
        A, _ = self.forward_prop(X)
        # Find the index of highest probability for each sample column
        max_indices = np.argmax(A, axis=0)
        # Create a one-hot prediction matrix matching shape of A
        preds = np.zeros_like(A)
        preds[max_indices, np.arange(X.shape[1])] = 1
        return preds, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        m = Y.shape[1]
        A_last = cache[f"A{self.__L}"]
        dZ = A_last - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache[f"A{layer-1}"]
            Wl = self.__weights[f"W{layer}"]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{layer}"] = (self.__weights[f"W{layer}"] -
                                           alpha * dW)
            self.__weights[f"b{layer}"] = (self.__weights[f"b{layer}"] -
                                           alpha * db)

            if layer > 1:
                A_prev_layer = cache[f"A{layer-1}"]
                if self.__activation == "sig":
                    dz = (A_prev_layer * (1 - A_prev_layer))
                elif self.__activation == "tanh":
                    dz = 1 - A_prev_layer ** 2
                dZ = np.dot(Wl.T, dZ) * dz

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the deep neural network """
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

        # Handle the evaluation and final recording at the last iteration
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
        """ Saves a deep neural network """
        if not (".pkl" in filename):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Loads a deep neural network """
        if not os.path.exists(filename):
            return None

        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    @property
    def L(self):
        return self.__L

    @property
    def activation(self):
        return self.__activation

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(z):
        """ Computes softmax over matrix column features safely """
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)
