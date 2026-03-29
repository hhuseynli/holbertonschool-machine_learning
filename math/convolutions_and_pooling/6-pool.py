#!/usr/bin/env python3
'''module documented'''
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''pooling in CNN'''
    h, kh = images.shape[1], kernel_shape[0]
    w, kw = images.shape[2], kernel_shape[1]
    sh, sw = stride
    c = images.shape[3]
    oh, ow = (int(np.floor((h - kh) / sh)) + 1,
              int(np.floor((w - kw) / sw)) + 1)

    output = np.zeros(shape=(images.shape[0], oh, ow, c))
    for row in range(oh):
        for col in range(ow):
            patch = images[:, row * sh: row * sh + kh,
                           col * sw: col * sw + kw, :]
            if mode == "max":
                output[:, row, col, :] = np.max(patch, axis=(1, 2))
            elif mode == "avg":
                output[:, row, col, :] = np.mean(patch, axis=(1, 2))
    return output
