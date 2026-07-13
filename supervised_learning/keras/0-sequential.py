#!/usr/bin/env python3
""" Build a sequential neural network"""

import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Args:
    nx -> number of input features
    layers -> lsit with number of nodes at each layer
    activations -> list with activations at each layer
    lambtha -> regularization parameter
    keep_prob -> probability of dropout

    Returns: the model
    """
    model = keras.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(
                keras.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=keras.regularizers.l2(lambtha),
                    input_shape=(nx,)
                )
            )
        else:
            model.add(
                keras.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=keras.regularizers.l2(lambtha)
                )
            )
        
        if i != len(layers) - 1:
            model.add(keras.layers.Dropout(1 - keep_prob))
    return model