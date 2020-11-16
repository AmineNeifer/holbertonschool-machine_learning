#!/usr/bin/env python3


""" contains forward, backward and baum_welch algos functs for hmm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model
    A = Transition
    B = Emission
    Pi = Initial
    O = Observation
    """
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros((N, T))
    for s in range(N):
        F[s, 0] = Initial[s] * Emission[s, Observation[0]]
    for t in range(1, T):
        obs_index = Observation[t]
        for s in range(N):
            F[s, t] = np.dot(F[:, t - 1], Transition[:, s]) * \
                Emission[s, Observation[t]]
    P = np.sum(F[:, T - 1], axis=0)
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    performs the backward algorithm for a hidden markov model
    A = Transition
    B = Emission
    Pi = Initial
    O = Observation
    """
    T = Observation.shape[0]
    N, M = Emission.shape

    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))
    for t in range(T - 2, -1, -1):
        for s in range(N):
            tmp = B[:, t + 1] * Emission[:, Observation[t + 1]]
            B[s, t] = np.dot(tmp, Transition[s, :])

    P = np.zeros((N, T))
    for s in range(N):
        P[s, 0] = Initial[s] * Emission[s, Observation[0]]
    P = np.sum(np.matmul(P.T, B[:, 0]))
    return P, B


def baum_welch(Observation, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model

    A = Transition
    B = Emission
    Pi = Initial
    O = Observation

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
