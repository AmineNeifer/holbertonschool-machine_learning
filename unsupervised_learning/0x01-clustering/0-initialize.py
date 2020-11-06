#!/usr/bin/env python3

""" contains a function that initializes centroids"""
import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray):
        return None
    if len(X.shape) != 2:
        return None
    if not isinstance(k, int):
        return None
    if k <= 0:
        return None
    n, d = X.shape
    high = np.amax(X, axis=0)
    low = np.amin(X, axis=0)
    cent = np.random.uniform(low, high, (k, d))
    return cent
