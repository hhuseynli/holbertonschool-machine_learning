#!/usr/bin/env python3
"Deep Neural Network"
import numpy as np


class DeepNeuralNetwork:
    """ A deep neural network consists of multiple hidden layers"""
    def __init__(self, nx, layers):
        if not (isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not (isinstance(layers, list)) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for layer in range(self.__L):
            if layers[layer] <= 0 or not (isinstance(layer, int)):
                raise TypeError("layers must be a list of positive integers")
            if layer == 0:
                prev = nx
            else:
                prev = layers[layer-1]
            W = (np.random.randn(layers[layer], prev) *
                 np.sqrt(2 / prev))
            b = np.zeros((layers[layer], 1))
            self.weights.update({f"W{layer+1}": W})
            self.weights.update({f"b{layer+1}": b})

    def forward_prop(self, X):
        """ Calculates the activation of each neuron
        Args:
            X -> ndarray (nx, m), where nx = features; m = examples
        Returns:
            Y, cache
        """
        prev = X
        for layer in range(self.__L+1):
            self.__cache[f"A{layer}"] = prev
            if layer != self.__L:
                z = self.__weights[f"W{layer+1}"] @ prev \
                    + self.__weights[f"b{layer+1}"]
                prev = self.sigmoid(z)
        return self.__cache[f"A{self.__L}"], self.__cache

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
