#!/usr/bin/env python3


""" contains a funct that returns optimum"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """optimum k"""
    if not isinstance(X, np.ndarray):
        return None, None
    if len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 0:
        return None, None
    if not isinstance(kmax, int) or kmin < 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax - kmin <= 2:
        return None, None
    results = []
    d_vars = []
    for i in range(kmin, kmax + 1):
        c, clss = kmeans(X, i, iterations)
        results.append((c, clss))
        d_vars.append(variance(X, c))
    for i in range(len(d_vars)):
        d_vars[i] = variance(X, results[0][0]) - d_vars[i]
    return results, d_vars
