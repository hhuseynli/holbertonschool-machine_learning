#!/usr/bin/env python3
""" Momentum implemented using numpy """

import tensorflow.keras as K


def create_momentum_op(alpha, beta1):
    """
    alpha is the learning rate

    beta1 is the momentum weight

    Returns: optimizer (keras)
    """
    return K.optimizers.SGD(learning_rate=alpha, momentum=beta1)
