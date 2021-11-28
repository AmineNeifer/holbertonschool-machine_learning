#!/usr/bin/env python3

import numpy as np

def backward(Observation, Emission, Transition, Initial):
    # add checks here
    T = Observation.shape[0]
    N, _ = Emission.shape
    
    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for t in range(T - 1, 0, -1):
        for s in range(N):
            B[s, t - 1] = np.sum(B[:, t] * Transition[s, :] * Emission[:, Observation[t]])
    return np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0]), B
