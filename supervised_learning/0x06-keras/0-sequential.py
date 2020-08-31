#!/usr/bin/env python3


""" contains build model function"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model with keras"""
    list_layer = []
    l2 = kernel_regularizer = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i < len(layers) - 1:
            if (i == 0):
                list_layer.append(
                    K.layers.Dense(
                        layers[i],
                        activation=activations[i],
                        input_shape=(
                            nx,
                        ),
                        kernel_regularizer=l2))
            else:
                list_layer.append(
                    K.layers.Dense(
                        layers[i],
                        activation=activations[i]),
                    kernel_regularizer=l2)
            list_layer.append(K.layers.Dropout(keep_prob))
        else:
            list_layer.append(K.layers.Dense(layers[i], kernel_regularizer=l2))

    model = K.Sequential(list_layer)
    return model
