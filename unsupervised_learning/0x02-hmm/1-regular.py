#!/usr/bin/env python3


""" regular funct"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain.

    @P square 2D np.ndarray, (n, n) representing the transition matrix
        -P[i, j] is the probability of transitioning from state i to state j
        -n is the number of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) representing the probability
    of being in a specific state after t iterations, or None on failure
    """
    np.warnings.filterwarnings('ignore')
    if not isinstance(P, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if (P < 0).any():
        return None
    n = P.shape[0]
    if not ((P > 0).all() and (P <= 1).all()):
        return None
    if not ((np.sum(P, axis=1) == np.ones((n))).all()).all():
        return None
    eig = np.linalg.eig(P)[0]
    if not (np.isclose(eig, 1)).any() and not (np.absolute(eig) <= 1).all():
        print(eig)
        print((eig == 1).any())
        return None
    if not (eig.dtype == float or eig.dtype == int):
        return None
    eye = np.eye(n)
    if np.allclose(P, eye) or np.allclose(P, eye.T):
        return None
    n = P.shape[0]
    a = np.eye(n) - P
    a = np.vstack((a.T, np.ones(n)))
    b = np.matrix([0] * n + [1]).T
    return np.linalg.lstsq(a, b)[0]
