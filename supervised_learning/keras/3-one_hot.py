#!/usr/bin/env python3
""" One-hot encoding """

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Use keras for one-hot encoding """
    return K.ops.one_hot(labels, classes)