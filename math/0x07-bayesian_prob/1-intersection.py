#!/usr/bin/env python3


""" Bayesian prob"""
import numpy as np
from math import factorial as fact


def comb(n, x):
    """ combination : nCx"""
    return fact(n) / (fact(x) * fact(n - x))


def likelihood(x, n, P):
    """ likelihood"""
    return comb(n, x) * P ** x * (1 - P) ** (n - x)


def intersection(x, n, P, Pr):
    """ intersection"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or len(Pr.shape) != 1:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not ((0 <= P).all() and (P <= 1).all()):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not ((0 <= Pr).all() and (Pr <= 1).all()):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    return likelihood(x, n, P) * Pr
