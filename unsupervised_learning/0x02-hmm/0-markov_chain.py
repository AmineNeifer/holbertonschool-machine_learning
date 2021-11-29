#!/usr/bin/env python3

import numpy as np


def markov_chain(P, s, t=1):
    """markov chain"""
    if type(P) is not np.ndarray:
        return None
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    n = P.shape[0]
    if type(s) is not np.ndarray:
        return None
    if s.shape != (1, n):
        return None
    if not np.sum(s) == 1:
        return None
    if t < 1 or t != int(t):
        return None
    Pt = np.linalg.matrix_power(P, t)
    return np.matmul(s, Pt)
