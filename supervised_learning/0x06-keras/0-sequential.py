#!/usr/bin/env python3


""" contains build model function"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model with keras"""
    l2 = K.regularizers.l2(lambtha)
    model = K.Sequential()
    for i in range(len(layers)):
        if i < len(layers) - 1:
            if (i == 0):
                model.add(
                    K.layers.Dense(
                        layers[i],
                        activation=activations[i],
                        input_shape=nx,
                        kernel_regularizer=l2))
            else:
                model.add(
                    K.layers.Dense(
                        layers[i],
                        activation=activations[i],
                        kernel_regularizer=l2))
            model.add(K.layers.Dropout(1 - keep_prob))
        else:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=l2))

    return model
