#!/usr/bin/env python3

import numpy as np
regular = __import__('1-regular').regular

def ergodic(P):
    """ergodic markov chain"""
    if type(P) is not np.ndarray:
        return False
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if not np.allclose(np.sum(P, axis=0), 1):
        return False
    if 1 in np.max(P, axis=0):
        return False
    if regular(P):
        return True
    # fix here
    return True

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.3],
                  [0.4, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.25, 0.3],
                  [0.25, 0.2, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[1, 0.25, 0, 0, 0],
                  [0, 0.75, 0, 0, 0],
                  [0, 0, 0.5, 0.3, 0.2],
                  [0, 0, 0.2, 0.5, .3],
                  [0, 0, 0.3, 0.2, 0.5]])
    e = np.array([[1, 0.25, 0, 0, 0],
                  [0, 0.75, 0.1, 0.1, 0.1],
                  [0, 0, 0.5, 0.2, 0.2],
                  [0, 0, 0.2, 0.5, .2],
                  [0, 0, 0.2, 0.2, 0.5]])
    print(ergodic(a.T))
    print(ergodic(b.T))
    print(ergodic(c.T))
    print(ergodic(d.T))
    print(ergodic(e.T))
