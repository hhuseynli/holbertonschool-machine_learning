#!/usr/bin/env python3
""" Using tf """

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """ Returns decayed learning rate """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        alpha,
        decay_step,
        decay_rate,
        staircase=True
    )
