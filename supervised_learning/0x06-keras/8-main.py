#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 8

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
import tensorflow.keras as K


# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model 

lib = np.load('./MNIST.npz')
X_train = lib['X_train']
X_train = X_train.reshape(X_train.shape[0], -1)
Y_train = one_hot(lib['Y_train'], 10)
X_valid = lib['X_valid']
X_valid = X_valid.reshape(X_valid.shape[0], -1)
Y_valid = one_hot(lib['Y_valid'], 10)
print("please")
model = build_model(784, [128, 64, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
optimize_model(model, 0.01, 0.99, 0.9)
history = train_model(model, X_train, Y_train, 64, 10, validation_data=(X_valid, Y_valid), verbose=False, save_best=True, filepath='0-test.h5').history
vl = np.around(history['val_loss'], 5)
va = history['val_acc']
i = np.argmin(vl)
print("something")
del model
model = K.models.load_model('0-test.h5')
l, a = model.evaluate(X_valid, Y_valid, verbose=False, batch_size=64)
print('anything')
assert(np.around(l, 5) == vl[i])
assert(np.around(a, 5) == np.around(va[i], 5))
print('OK', end='')
