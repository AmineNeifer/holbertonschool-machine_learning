#!/usr/bin/env python3

""" functions to find kmean"""
import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray):
        return None
    if len(X.shape) != 2:
        return None
    if not isinstance(k, int):
        return None
    if k <= 0:
        return None
    n, d = X.shape
    high = np.amax(X, axis=0)
    low = np.amin(X, axis=0)
    cent = np.random.uniform(low, high, (k, d))
    return cent


def closest_centroid(points, centroids):
    """
    returns an array containing the idx to the nearest centroid for each nt
    """
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def kmeans(X, k, iterations=1000):
    """ kmeans algorithm"""
    cent = initialize(X, k)
    if cent is None:
        return None, None
    if not isinstance(iterations, int):
        return None, None
    if iterations <= 0:
        return None, None
    closest = closest_centroid(X, cent)
    for i in range(iterations):
        cp = np.copy(cent)
        for j in range(k):
            if X[np.where(closest == j)].size == 0:
                cent[j] = initialize(X, 1)
            else:
                cent[j] = X[np.where(closest == j)].mean(axis=0)
        closest = closest_centroid(X, cent)
        if np.array_equal(cp, cent):
            break

    return cent, closest
