#!/usr/bin/env python3

import numpy as np

def viterbi(Observation, Emission, Transition, Initial):
    # add checks here
    T = Observation.shape[0]
    N, _ = Emission.shape

    V = np.zeros((N, T))
    B = np.zeros((N, T))
    for s in range(N):
        V[s, 0] = Initial[s] * Emission[s, Observation[0]]
    for t, o in enumerate(Observation):
        if t == 0:
            continue
        for s in range(N):
            temp = V[:, t - 1] * Transition[:, s] * Emission[s, o]
            V[s, t] = np.max(temp)
            B[s, t] = np.argmax(temp)
    prob = np.max(V[:, T - 1])
    pointer = np.argmax(V[:, T - 1])
    path = [pointer]
    for t in range(T - 1, 0, -1):
        p = int(B[pointer, t])
        path.append(p)
        pointer = p
    return path[::-1], prob
