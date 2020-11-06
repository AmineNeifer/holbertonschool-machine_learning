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
