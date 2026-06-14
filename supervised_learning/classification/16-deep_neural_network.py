#!/usr/bin/env python3
"Deep Neural Network"
import numpy as np


class DeepNeuralNetwork:
    """ A deep neural network consists of multiple hidden layers"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layer must be a list of positive integers")
        if (np.array(layers) > 0).sum() != len(layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = dict()
        self.weights = dict()

        for layer in range(self.L):
            if layer == 0:
                prev = nx
            else:
                prev = layers[layer-1]
            W = np.random.randn(layers[layer], prev) * np.sqrt(2 / prev)
            b = np.zeros((layers[layer], 1))

            self.weights[f"W{layer+1}"] = W
            self.weights[f"b{layer+1}"] = b
