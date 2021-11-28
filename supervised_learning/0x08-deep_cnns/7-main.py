#!/usr/bin/env python3

import tensorflow.keras as K
densenet121 = __import__('7-densenet121').densenet121

model = densenet121(32, 0.5)
model.summary()
