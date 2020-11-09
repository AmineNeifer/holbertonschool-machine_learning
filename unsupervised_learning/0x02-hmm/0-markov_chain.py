#!/usr/bin/env python3


""" contains markov_chaine func"""
import numpy as np


def markov_chain(P, s, t=1):
    """ 
    Determines the probability of a markov chain being in a 
    particular state after a specified number of iterations

    @P square 2D np.ndarray, (n, n) representing the transition matrix
        -P[i, j] is the probability of transitioning from state i to state j
        -n is the number of states in the markov chain
    @s np.ndarray, (1, n) represents the prob of starting in each state
    @t number of iterations that the markov chain has been through
    
    Returns: a numpy.ndarray of shape (1, n) representing the probability
    of being in a specific state after t iterations, or None on failure
    """
    if not isinstance(P, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray):
        return None
    if len(s.shape) != 2 or s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int):
        return None
    if t <= 0:
        return None
    for i in range(t):
        s = np.matmul(s, P)
    return s
