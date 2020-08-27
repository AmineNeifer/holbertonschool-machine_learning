#!/usr/bin/env python3


""" contains dropout forward prop funct"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Returns: a dictionary containing the outputs of
    each layer and the dropout mask used on each layer"""
    m = X.shape[1]
    cache = {'A0': X}
    for i in range(L):
        A = 'A' + str(i + 1)
        A_prev = 'A' + str(i)
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        Z = np.matmul(W, cache[A_prev]) + b
        if i == L - 1:
            t = np.exp(Z)
            cache[A] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache[A] = 2 / (1 + np.exp(-2 * Z)) - 1
            d = np.random.rand(
                cache[A].shape[0],
                cache[A].shape[1]) < keep_prob
            cache['D' + str(i + 1)] = np.where(d, 1, 0)
            cache[A] *= d
            cache[A] /= keep_prob
    return cache
