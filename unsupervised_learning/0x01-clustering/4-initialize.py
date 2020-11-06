#!/usr/bin/env python3


"""initializing GMM"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ initializes variables for a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray):
        return None, None, None
    if len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    d = X.shape[1]
    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    new = np.eye(d, d)
    S = np.zeros((k, d, d))
    S[:] = new
    return pi, m, S
