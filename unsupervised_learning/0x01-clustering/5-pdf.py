#!/usr/bin/env python3

""" prob density function of a Gaussian dist"""
import numpy as np


def pdf(X, m, S):
    """ pdf of Gaussian dist"""
    if not isinstance(X, np.ndarray):
        return None
    if len(X.shape) != 2:
        return None
    n, d = X.shape
    if not isinstance(m, np.ndarray):
        return None
    if len(m.shape) != 1 or m.shape != (d,):
        return None
    if not isinstance(S, np.ndarray):
        return None
    if len(S.shape) != 2 or S.shape != (d, d):
        return None
    x = X
    E = np.linalg.det(S)
    inv = np.linalg.inv(S)
    denom = 1 / ((((2 * np.pi) ** (d)) * E) ** 0.5)
    f_matmul = np.matmul((x - m), inv)
    nominator = np.exp(-(0.5) * np.matmul(f_matmul, (x - m).T))
    p = denom * nominator
    p = p * np.eye(p.shape[0], p.shape[1])
    p = p[p != 0]
    p = p.flatten()
    p = np.where(p >= 1e-300, p, 1e-300)
    return p
