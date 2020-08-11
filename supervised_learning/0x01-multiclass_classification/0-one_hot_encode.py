#!/usr/bin/env python3


""" contains one_hot_encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """ converts a numeric label vector into a one-hot matrix"""
    if Y is None or classes is None:
        return None
    if type(classes) is not int:
        return None
    if classes < 0:
        return None
    if Y.max() >= classes:
        return None
    try:
        new = np.zeros((Y.shape[0], classes))
        for i in range(len(Y)):
            new[i][Y[i]] = 1
        return new.T
    except IndexError:
        return None
