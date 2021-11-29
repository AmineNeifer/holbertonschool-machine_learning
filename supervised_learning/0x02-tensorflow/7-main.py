#!/usr/bin/env python3

import numpy as np

evaluate = __import__('7-evaluate').evaluate

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

np.random.seed(0)
m1 = np.random.randint(10, 50)
m2 = np.random.randint(m1, 100)
c = 10

lib = np.load('MNIST.npz')
X = lib['X_train'][m1:m2].reshape((m2 - m1, -1))
Y = one_hot(lib['Y_train'][m1:m2], c)

print(evaluate(X, Y, './model.ckpt'))
