#!/usr/bin/env python3
""" Plot x^3 """
import numpy as np
import matplotlib.pyplot as plt


def line():
    """ Set bounds to 0<=x<=10 and color to red """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 10)
    plt.plot(y, "r")
    plt.show()
