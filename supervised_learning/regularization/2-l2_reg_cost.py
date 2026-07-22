#!/usr/bin/env python3
""" L2 Regularization (tf)"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """ Returns a tensor containing total cost for each layer """
    reg_loss = [tf.add_n(lay.losses) for lay in model.layers if lay.losses]
    return cost + reg_loss
