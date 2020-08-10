#!/usr/bin/env python3


import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix"""
    new = np.zeros((Y.shape[0], classes))
    for i in range(len(Y)):
        new[i][Y[i]] = 1
    return new.T
