#!/usr/bin/env python3

import tensorflow.keras as K
resnet50 = __import__('4-resnet50').resnet50

model = resnet50()
model.summary()