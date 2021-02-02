#!/usr/bin/env python3


""" expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray):
        return None, None
    if len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if not isinstance(pi, np.ndarray):
        return None, None
    if len(pi.shape) != 1:
        return None, None
    k = pi.shape[0]
    if not np.isclose(pi.sum(), 1):
        return None, None
    if not isinstance(m, np.ndarray):
        return None, None
    if len(m.shape) != 2 or m.shape != (k, d):
        return None, None
    if not isinstance(S, np.ndarray):
        return None, None
    if len(S.shape) != 3 or S.shape != (k, d, d):
        return None, None
    y = np.zeros([k, n])
    for i in range(k):
        y[i] = pdf(X, m[i], S[i]) * pi[i]

    likelihood = np.log(y.sum(axis=0)).sum()
    y /= y.sum(axis=0)
    return y, likelihood
