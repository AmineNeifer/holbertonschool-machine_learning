#!/usr/bin/env python3


""" contains one hot decoder function"""
import numpy as np


def one_hot_decode(one_hot):
    """ that converts a one-hot matrix into a vector of labels"""
    if one_hot is None:
        return None
    m = one_hot.shape[0]
    new = np.ndarray((m,))
    for i in range(m):
        new[i] = int(np.argmax(one_hot.T[:][i]))
    return new
