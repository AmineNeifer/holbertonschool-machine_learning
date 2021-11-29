#!/usr/bin/env python3

import numpy as np

def absorbing(P):
    """absorbing markov chain"""
    if type(P) is not np.ndarray:
        return False
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if not np.allclose(np.sum(P, axis=1), 1):
        return False
    n = P.shape[0]
    for j in range(n):
        if P[j, j] != 1:
            i = j
            break
    else:
        return True
    if i == 0:
        return False
    if (1 in np.max(P[i:, :], axis=0)) or (np.sum(P[:i, i:]) != 0):
        return False
    Q = P[i:, i:]
    R = P[i:, :i]
    t = Q.shape[0]
    IQ = np.eye(t) - Q
    if np.linalg.det(IQ) == 0:
        return False
    N = np.linalg.inv(IQ)
    B = np.matmul(N, R)
    if 0 in np.sum(B, axis=1):
        return False
    return True
