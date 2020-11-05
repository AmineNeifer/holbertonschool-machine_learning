#!/usr/bin/env python3


import numpy as np

def initialize(X, k):
    """ initializes cluster centroids for K-means"""
    n, d = X.shape
    high = np.amax(X, axis=0)
    low = np.amin(X, axis=0)
    cent = np.random.uniform(low, high, (k, d))
    return cent
def kmeans(X, k, iterations=1000):
    """ kmeans algorithm"""
    cent = initialize(X, k)
    assigned_cent = np.zeros((X.shape[0],), dtype=int)
    for i in range(X.shape[0]):
        distance = np.linalg.norm(X[i] - cent, axis=1)
        assigned_cent[i] = np.argmin(distance)
    for i in range(cent.shape[0]):
        if X[np.where(assigned_cent == i)].size != 0:
            cent[i] = X[np.where(assigned_cent == i)].mean(axis=0)
    return cent, assigned_cent

    