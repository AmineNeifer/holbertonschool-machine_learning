#!/usr/bin/env python3

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    network = K.models.Sequential()
    network.add(K.layers.Dense(layers[0], activation=activations[0], input_shape=(
        nx,), kernel_initializer=K.initializers.he_normal(), kernel_regularizer=K.regularizers.l2(lambtha)))
    if len(layers) > 1:
        network.add(K.layers.Dropout(1 - keep_prob))
        for i, layer in enumerate(layers[1:-1]):
            network.add(K.layers.Dense(layer, activation=activations[i+1], kernel_initializer=K.initializers.he_normal(
            ), kernel_regularizer=K.regularizers.l2(lambtha)))
            network.add(K.layers.Dropout(1 - keep_prob))
        network.add(K.layers.Dense(layers[-1], activation=activations[-1],
                    kernel_initializer=K.initializers.he_normal(), kernel_regularizer=K.regularizers.l2(lambtha)))
    return network
