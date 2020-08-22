#!/usr/bin/env python3


""" contains normalization_constatnts funct"""
import numpy as np


def normalization_constants(X):
    """ Returns: the mean and standard deviation, respectively"""
    m = X.shape[0]
    mean = np.sum(X / m, axis=0)
    stddev = np.sqrt(np.sum(((X - mean) ** 2), axis=0) / m)
    return mean, stddev
