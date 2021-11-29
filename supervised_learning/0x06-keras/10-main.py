#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 10

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
load_weights = __import__('10-weights').load_weights

model = build_model(784, [128, 64, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
model.save_weights('./1-test.h5')
del model
model = build_model(784, [128, 64, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
optimize_model(model, 0.01, 0.99, 0.9)
load_weights(model, './1-test.h5')
print(model.get_weights())
