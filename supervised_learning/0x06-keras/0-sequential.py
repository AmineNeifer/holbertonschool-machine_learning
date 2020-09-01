#!/usr/bin/env python3


""" contains build model function"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model with keras tf.v01"""
    model = K.Sequential()
    l2 = K.regularizers.l2(lambtha)
    model.add(K.layers.Dense(
        layers[0],
        activation=activations[0],
        input_shape=(nx,),
        kernel_regularizer=l2))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(
            K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=l2))
    return model
