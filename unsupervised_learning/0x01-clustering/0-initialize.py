#!/usr/bin/env python3

""" contains a function that initializes centroids"""
import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means"""
    n, d = X.shape
    high = np.amax(X, axis=0)
    low = np.amin(X, axis=0)
    cent = np.random.uniform(low, high, (k, d))
    return cent
