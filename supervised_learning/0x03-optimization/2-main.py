#!/usr/bin/env python3

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data

np.random.seed(2)

m, nx, ny = np.random.randint(10, 100, 3)
X = np.random.randn(m, nx)
Y = np.random.randn(m, ny)
print(shuffle_data(X, Y))