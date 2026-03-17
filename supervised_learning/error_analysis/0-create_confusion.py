#!/usr/bin/env python3
""" Make a confusion matrix """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ input: labels and predictions (m, classes) shaped
    output: confusion matrix(classes, classes) shaped """
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    actual = np.argmax(labels, axis=1)
    prediction = np.argmax(logits, axis=1)

    np.add.at(confusion, (actual, prediction), 1)
    return confusion
