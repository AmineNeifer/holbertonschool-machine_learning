#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt


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


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest == k].mean(axis=0)
                     for k in range(centroids.shape[0])])


def kmeans(X, k, iterations=1000):
    """ kmeans algorithm"""
    c = initialize(X, k)
    i = 0

    for i in range(10):
        closest = closest_centroid(X, cent)
        if np.isnan(cent).any():
            cent = initialize(X, k).copy()
        else:
            cent = move_centroids(X, closest, cent)
        plt.scatter(X[:, 0], X[:, 1], s=10, c=closest)
        plt.scatter(cent[:, 0], cent[:, 1], s=50, marker='*', c=list(range(5)))
        plt.show()

    return cent, closest
