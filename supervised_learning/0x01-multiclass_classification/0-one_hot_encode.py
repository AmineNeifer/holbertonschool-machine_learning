#!/usr/bin/env python3


import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix"""
    if Y is None or classes is None:
        return None
    if type(classes) is not int:
        return None
    if classes <= 0:
        return None
    if Y.max() > classes:
        return None
    new = np.zeros((Y.max(), Y.shape[0]))
    for i in range(Y.max()):
        new[i][Y[i]] = 1
    return new.T
