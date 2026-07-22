#!/usr/bin/env python3
""" Create a regularization layer """

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Create an L2 layer using tf """
    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_avg'
        ),
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )
    return layer(prev)
