#!/usr/bin/env python3
"""Predicts y values duhh"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ Predict activated outputs"""
    return network.predict(data, verbose=verbose)
