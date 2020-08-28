#!/usr/bin/env python3


""" contains dropout gradient descent funct"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights and biases of a neural network
    using gradient descent with dropout"""
    m = Y.shape[1]
    weis = weights.copy()

    for i in range(L - 1, -1, -1):
        prev_A = cache['A' + str(i)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        A = cache['A' + str(i + 1)]
        if i == L - 1:
            dz = A - Y
        else:
            D = cache['D' + str(i + 1)]
            A = A * D / keep_prob
            dz = np.matmul(weis['W' + str(i + 2)].T, dz) * (1 - (A * A))
        dW = np.matmul(dz, prev_A.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i + 1)] = weights["W" + str(i + 1)] - (alpha * dW)
        weights['b' + str(i + 1)] = weights["b" + str(i + 1)] - (alpha * db)
