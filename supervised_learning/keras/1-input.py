#!/usr/bin/env python3
""" Build a sequential neural network"""

import tensorflow.keras as K


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
    inputs = K.Input(shape=(nx,))
    a = inputs
    for i in range(len(layers)):
        a = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
        )(a)
        if i != len(layers) - 1:
            a = K.layers.Dropout(1 - keep_prob)(a)

    model = K.Model(inputs=inputs, outputs=a)
    return model
