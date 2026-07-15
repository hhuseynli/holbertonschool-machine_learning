#!/usr/bin/env python3
""" Using numpy """

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Learning rate decay """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
