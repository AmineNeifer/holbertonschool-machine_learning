#!/usr/bin/env python3

""" prob density function of a Gaussian dist"""
import numpy as np


def pdf(X, m, S):
    """ pdf of Gaussian dist"""
    x = X
    E = np.linalg.det(S)
    inv = np.linalg.inv(S)
    d = m.shape[0]
    denom = 1 / ((((2 * np.pi) ** (d)) * E) ** 0.5)
    f_matmul = np.matmul((x - m), inv)
    nominator = np.exp(-(0.5) * np.matmul(f_matmul, (x - m).T))
    p = denom * nominator
    p = p * np.eye(p.shape[0], p.shape[1])
    p = p.flatten()
    p = p[p > 1e-300]
    return p
