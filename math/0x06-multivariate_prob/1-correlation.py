#!/usr/bin/env python3
""" contains correlation function"""
import numpy as np


def correlation(C):
    """ returns correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlation = C / outer_v
    correlation[C == 0] = 0
    return correlation
