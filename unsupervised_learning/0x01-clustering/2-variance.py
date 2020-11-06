#!/usr/bin/env python3


""" varaince intra-clusters"""
import numpy as np

def closest_centroid(points, centroids):
    """
    returns an array containing the idx to the nearest centroid for each nt
    """
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def variance(X, C):
    """ total variance inntra-cluster"""
    if not isinstance(X, np.ndarray):
        return None
    if len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray):
        return None
    if len(C.shape) != 2 or C.shape[1] != X.shape[1]:
        return None
    n, d = X.shape
    k, _ = C.shape
    closest = closest_centroid(X, C)
    ssw = np.linalg.norm(X - C[closest]) ** 2
    return ssw
