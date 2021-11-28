#!/usr/bin/env python3

import tensorflow.keras as K
optimize_model = __import__('2-optimize').optimize_model


def build_model(nx, layers, activations, lambtha, keep_prob):
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i, layer in enumerate(layers[:-1]):
        x = K.layers.Dense(layer, activation=activations[i], kernel_initializer=K.initializers.he_normal(), kernel_regularizer=K.regularizers.l2(lambtha))(x)
        x = K.layers.Dropout(1 - keep_prob)(x)
    predictions = K.layers.Dense(layers[-1], activation=activations[-1], kernel_initializer=K.initializers.he_normal(), kernel_regularizer=K.regularizers.l2(lambtha))(x)
    network = K.Model(inputs=inputs, outputs=predictions)
    return network

model = build_model(200, [100, 50, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
optimize_model(model, 0.01, 0.99, 0.9)
print(model.loss)
opt = model.optimizer
print(opt.__class__)
print(tuple(map(lambda x: x.numpy(),(opt.lr, opt.beta_1, opt.beta_2))))

