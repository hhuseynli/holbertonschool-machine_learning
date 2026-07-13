#!/usr/bin/env python3
"""12-test.py"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Evaluate model performance"""
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
