#!/usr/bin/env python3


""" exp max algorithm"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ EM Algorithm"""
    if not isinstance(X, np.ndarray):
        return None, None, None, None, None
    if len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
