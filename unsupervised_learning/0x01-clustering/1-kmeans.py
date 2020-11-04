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
    