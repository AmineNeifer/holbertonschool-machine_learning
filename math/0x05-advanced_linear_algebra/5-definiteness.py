#!/usr/bin/env python3

""" contains definiteness funct"""
import numpy as np


def definiteness(matrix):
    """ returns the definitenetss of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0:
        return None
    if len(matrix.shape) != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix != matrix.T:
        return None
    x = matrix
    if np.all(np.linalg.eigvals(x) > 0):
        return "Positive definite"
    elif np.all(np.linalg.eigvals(x) >= 0):
        return "Positive semi-definite"
    elif np.all(np.linalg.eigvals(x) < 0):
        return "Negative definite"
    elif np.all(np.linalg.eigvals(x) <= 0):
        return "Negative semi-definite"
    elif np.all(np.linalg.eigvals(x) != 0):
        return "Indefinite"
    else:
        return None
