#!/usr/bin/env python3

import tensorflow.keras as K
build_model = __import__('1-input').build_model

model = build_model(200, [10], ['tanh'], 0.01, 0.6)
model.summary()
