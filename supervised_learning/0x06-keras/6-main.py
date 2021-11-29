#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 7

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
train_model = __import__('6-train').train_model

lib = np.load('./MNIST.npz')
X_train = lib['X_train']
X_train = X_train.reshape(X_train.shape[0], -1)
Y_train = one_hot(lib['Y_train'], 10)
model = build_model(784, [128, 64, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
optimize_model(model, 0.01, 0.99, 0.9)
history = train_model(model, X_train, Y_train, 64, 30, early_stopping=True, patience=1, verbose=False).history
print(history.get('val_loss'))
loss = history['loss']
print(len(loss) == 30)
