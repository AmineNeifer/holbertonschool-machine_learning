#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
predict = __import__('13-predict').predict

def build_model(nx, layers, activations, lambtha, keep_prob):
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i, layer in enumerate(layers[:-1]):
        x = K.layers.Dense(layer, activation=activations[i], kernel_initializer=K.initializers.he_normal(), kernel_regularizer=K.regularizers.l2(lambtha))(x)
        x = K.layers.Dropout(1 - keep_prob)(x)
    predictions = K.layers.Dense(layers[-1], activation=activations[-1], kernel_initializer=K.initializers.he_normal(), kernel_regularizer=K.regularizers.l2(lambtha))(x)
    network = K.Model(inputs=inputs, outputs=predictions)
    return network

def optimize_model(network, alpha, beta1, beta2):
    network.compile(optimizer=K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2),
                    loss='categorical_crossentropy', metrics=['accuracy'])

np.random.seed(13)
tf.random.set_seed(0)
lib = np.load('./MNIST.npz')
X_valid = lib['X_valid'][:100].reshape(100, -1)
model = build_model(784, [128, 64, 10], ['tanh', 'sigmoid', 'softmax'], 0.01, 0.6)
optimize_model(model, 0.01, 0.99, 0.9)
print(predict(model, X_valid))
