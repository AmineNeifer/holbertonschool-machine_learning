#!/usr/bin/env python3


""" contains maximization funct"""
import numpy as np


def maximization(X, g):
    """M step for E: algorithm for a GMM"""
    if not isinstance(X, np.ndarray):
        return None, None, None
    if len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray):
        return None, None, None
    if len(g.shape) != 2:
        return None, None, None
    m = np.sum(np.matmul(g, X), axis=0) / np.sum(g, axis=1)
    pi = 1 / (n * np.sum(g, axis=1))
    S = np.zeros((k, d, d))
    for i in range(k):
        S[i] = np.matmul(g[i].reshape(1, n) * (X - m[i]).T, (X - m[i]))
        S[i] /= np.sum(g, axis=1)[i]
    return pi, m, S
