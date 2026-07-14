#!/usr/bin/env python3
""" Normalization """

import numpy as np


def normalization_constants(X):
    return X.mean(axis=0), X.std(axis=0)
