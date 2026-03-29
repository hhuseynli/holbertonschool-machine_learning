#!/usr/bin/env python3
'''module documented'''
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''same convolution'''
    h, kh = images.shape[1], kernel.shape[0]
    w, kw = images.shape[2], kernel.shape[1]
    ph, pw = padding
    oh, ow = ((h - kh + 2 * ph + 1),
              (w - kw + 2 * pw + 1))
    images = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    constant_values=(0))
    output = np.zeros(shape=(images.shape[0], oh, ow))
    for row in range(oh):
        for col in range(ow):
            patch = images[:, row:row+kh, col:col+kw]
            output[:, row, col] = np.sum(patch * kernel, axis=(1, 2))
    return output
