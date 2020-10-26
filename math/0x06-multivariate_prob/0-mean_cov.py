#!/usr/bin/env python3

""" contains a function that calculates both mean and covariance"""
import numpy as np


def mean_cov(X):
    """ returns mean and covariance"""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    return mean, np.matmul((X - mean).T, (X - mean)) / (X.shape[0] - 1)
