#!/usr/bin/env python3


""" bic funct"""
import numpy as np
expectation_maximization = __import__('7-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ finds best number of clusters for GMM"""
    f not isinstance(X, np.ndarray):
        return None, None
    if len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 0:
        return None, None, None, None
    if not isinstance(kmax, int) or kmin < 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
