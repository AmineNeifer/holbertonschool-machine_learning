#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop

np.random.seed(7)
l = np.random.randint(2, 10)
sizes = np.random.randint(10, 1000, l + 1)
m = np.random.randint(1000, 10000)
weights = {}
for i in range(l):
    weights['W' + str(i + 1)] = np.random.randn(sizes[i + 1], sizes[i])
    weights['b' + str(i + 1)] = np.random.randn(sizes[i + 1], 1)
X = np.random.randn(sizes[0], m)
kp = np.random.uniform(0.5)
cache = dropout_forward_prop(X, weights, l, kp)
for k, v in sorted(cache.items()):
    print(k, v)
