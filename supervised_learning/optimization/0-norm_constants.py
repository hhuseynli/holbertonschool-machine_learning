#!/usr/bin/env python3
""" Normalization """

import numpy as np


def normalization_constants(X):
    """ Returns the mean and std of each feature"""
    return X.mean(axis=0), X.std(axis=0)
