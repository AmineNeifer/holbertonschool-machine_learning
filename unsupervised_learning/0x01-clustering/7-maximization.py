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
