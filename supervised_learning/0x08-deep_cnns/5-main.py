#!/usr/bin/env python3

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block

X1 = K.Input(shape=(56, 56, 64))
Y1, _ = dense_block(X1, 64, 32, 6)
model1 = K.models.Model(inputs=X1, outputs=Y1)
for layer in model1.layers:
    if type(layer) is K.layers.Conv2D:
        for k, v in sorted(layer.kernel_initializer.__dict__.items()):
            if k == "_random_generator":
                print(k, type(v))
            else:
                print(k, v)
X2 = K.Input(shape=(56, 56, 64))
Y2, _ = dense_block(X2, 64, 32, 6)
model2 = K.models.Model(inputs=X2, outputs=Y2)
for layer in model2.layers:
    if type(layer) is K.layers.Conv2D:
        for k, v in sorted(layer.kernel_initializer.__dict__.items()):
            if k == "_random_generator":
                print(k, type(v))
            else:
                print(k, v)
