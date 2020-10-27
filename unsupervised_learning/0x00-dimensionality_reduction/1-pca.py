#!/usr/bin/env python3
""" pca """
import numpy as np


def pca(X, ndim):
    """ pca yehi"""
    X_m = X - np.mean(X, axis=0)
    m = X.shape[0]
    u, s, vh = np.linalg.svd(X_m)
    v = np.cumsum(s) / np.sum(s)
    W = vh.T[:, :ndim]
    T = np.matmul(X_m, W)
    return T
