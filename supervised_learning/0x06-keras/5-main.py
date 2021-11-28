#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 6

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('5-train').train_model

m1 = np.random.randint(100, 200)
m2 = np.random.randint(m1, 300)
bs = 2 ** np.random.randint(5, 9)
lib = np.load('./MNIST.npz')
X_train = lib['X_train'][m1:m2].reshape(m2 - m1, -1)
Y_train = one_hot(lib['Y_train'][m1:m2], 10)
model = build_model(784, [128, 64, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
optimize_model(model, 0.01, 0.99, 0.9)
history = train_model(model, X_train, Y_train, bs, 5, verbose=False).history
for k, v in sorted(history.items()):
    print(k, np.round(v, 5))
