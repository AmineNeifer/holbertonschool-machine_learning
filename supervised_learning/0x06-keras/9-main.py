#!/usr/bin/env python3
"""
Main file
"""

# Force Seed - fix for Keras
SEED = 9

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
load_model = __import__('9-model').load_model

model = build_model(784, [128, 64, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
model.save('./1-test.h5')
config = model.get_config()
del model
model = load_model('./1-test.h5')
print(model.get_weights())
print(model.get_config() == config)
