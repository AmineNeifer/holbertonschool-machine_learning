#!/usr/bin/env python3


""" contains one hot decoder function"""
import numpy as np


def one_hot_decode(one_hot):
    """ that converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray:
        return None
    if (one_hot > 1).all() or (one_hot < 0).all():
        return None
    if np.sum(one_hot) != one_hot.shape[1]:
        return None
    return np.argmax(one_hot, axis=0)
