#!/usr/bin/env python3


""" contains build model funct that uses Input instead of sequential"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build a model using Input class"""
    l2 = kernel_regularizer = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    dense = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=l2)
    x = dense(inputs)
    x = K.layers.Dropout(keep_prob)(x)
    for i in range(1, len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=l2)(x)
        if (i == len(layers) - 1):
            break
        x = K.layers.Dropout(keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
