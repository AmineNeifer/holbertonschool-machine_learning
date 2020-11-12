#!/usr/bin/env python3

""" contains absorbing function"""
import numpy as np


def look_for_ones(P):
    """
    looks for a column in which there is 1
    Return:
        True if there's a column with number 1 and sum over the column is >=1
        False otherwise
    """
    mat = P.copy()
    for col in mat.T:
        if 1 in col:
            return True
    return False


def absorbing_helper(P):
    """
    removes the part where we have the I O (from standard)
    checks if it can lead to total absorbing
    """
    mat = P.copy()
    for i in range(1, len(mat)):
        if not ((mat[:i, :i] == np.eye(i)).all()).all():
            break
    i -= 1
    res = mat[i:, i:]
    for r in res.T:
        nbr = np.where(r == 0, 1, 0)
        nbr = nbr.sum()
        if (nbr == (len(res) - 1)).all():
            return False
    return not (mat[i:, :i] == 0).all()


def absorbing(P):
    """ determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray):
        return False
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False
    if not look_for_ones(P):
        return False
    if ((P == np.eye(P.shape[0])).all()).all():
        return True
    return absorbing_helper(P)
