#!/usr/bin/env python3

import numpy as np


def regular(P):
    """regular markov chain"""
    if type(P) is not np.ndarray:
        return None
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    eigvals, eigvecs = np.linalg.eig(P.T)
    one = np.argwhere(np.isclose(eigvals, 1))
    if len(one) != 1:
        return None
    steady = eigvecs[:, one[0]] / np.sum(eigvecs[:, one[0, 0]])
    if 0 in steady:
        return None
    return steady.T
