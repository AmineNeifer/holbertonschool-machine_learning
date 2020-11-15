#!/usr/bin/env python3

""" viterbi algo HMM"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    calculates the most likely sequence of hidden states for a hmm
    """
    if not isinstance(Observation, np.ndarray):
        return None, None
    if len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray):
        return None, None
    if len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray):
        return None, None
    if Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray):
        return None, None
    if Initial.shape != (N, 1):
        return None, None
    D = np.zeros((N, T))
    E = np.zeros((N, T - 1)).astype(np.int32)
    for s in range(N):
        D[s, 0] = Initial[s] * Emission[s, Observation[0]]
    for n in range(1, T):
        for i in range(N):
            tmp = np.multiply(Transition[:, i], D[:, n - 1])
            D[i, n] = np.max(tmp) * Emission[i, Observation[n]]
            E[i, n - 1] = np.argmax(tmp)
    S_opt = np.zeros(T).astype(np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(T - 2, -1, -1):
        S_opt[n] = E[int(S_opt[n + 1]), n]

    omx = D[0, T - 1]
    for i in range(1, N):
        if D[i, T - 1] > omx:
            omx = D[i, T - 1]

    return S_opt, omx
