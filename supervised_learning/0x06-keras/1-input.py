#!/usr/bin/env python3


""" contains build model funct that uses Input instead of sequential"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build a model using Input class"""
    inputs = K.Input(shape=(nx,))
    l2 = kernel_regularizer = K.regularizers.l2(lambtha)
    dense = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=l2)
    x = dense(inputs)
    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=l2)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
