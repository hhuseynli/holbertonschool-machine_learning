#!/usr/bin/env python3
""" RMSprop using tf"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """ Returns RMSProp optimizer """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
