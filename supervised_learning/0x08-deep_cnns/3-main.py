#!/usr/bin/env python3

import tensorflow.keras as K
projection_block = __import__('3-projection_block').projection_block

X1 = K.Input(shape=(112, 112, 256))
Y1 = projection_block(X1, [64, 64, 256])
model1 = K.models.Model(inputs=X1, outputs=Y1)
model1.summary()
X2 = K.Input(shape=(56, 56, 512))
Y2 = projection_block(X2, [128, 128, 512])
model2 = K.models.Model(inputs=X2, outputs=Y2)
model2.summary()
