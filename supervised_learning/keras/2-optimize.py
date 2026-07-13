#!/usr/bin/env python3
""" Build a sequential neural network"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Compiles model with loss, optimizer and metrics
    Args:
    network -> model to optimize
    alpha -> learning rate
    beta1, beta2 -> first and second adam optimizers respectively

    Returns: None
    """
    network.compile(
        loss=K.losses.CategoricalCrossentropy(),
        optimizer=K.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1,
                                    beta_2=beta2),
        metrics=['accuracy']
    )
