#!/usr/bin/env python3


""" contains l2 reg gradient descent funct"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network using gradient descent with L2 regularization"""
    m = Y.shape[1]
    new_ws = weights.copy()

    """dZ3 = cache['A3'] - Y
    dW3 = np.matmul(dZ3, cache['A2'].T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    weights['W3'] = new_ws['W3'] - (alpha * dW3) + lambtha * new_ws['W3'] / m
    weights['b3'] = new_ws['b3'] - (alpha * db3)

    dZ2 = np.matmul(new_ws['W3'].T, dZ3) * (1 - (cache['A2'] ** 2))
    dW2 = np.matmul(dZ2, cache['A1'].T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    weights['W2'] = new_ws['W2'] - (alpha * dW2) + lambtha * new_ws['W2'] / m
    weights['b2'] = new_ws['b2'] - (alpha * db2)

    dZ1 = np.matmul(new_ws['W2'].T, dZ2) * (1 - (cache['A1'] ** 2))
    dW1 = np.matmul(dZ1 , cache['A0'].T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    weights['W1'] = new_ws['W1'] - (alpha * dW1) + lambtha * new_ws['W1'] / m
    weights['b1'] = new_ws['b1'] - (alpha * db1)"""
    for i in range(L - 1, -1, -1):
        prev_A = cache['A' + str(i)]
        A = cache['A' + str(i + 1)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        if i == L - 1:
            dz = A - Y
        else:
            dz = np.matmul(weights['W' + str(i + 2)].T, dz) * (1 - (A * A))
        dW = np.matmul(dz, prev_A.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights['W' + str(i + 1)] = W - (alpha * dW) + lambtha * W / m
        weights['b' + str(i + 1)] = b - (alpha * db)
