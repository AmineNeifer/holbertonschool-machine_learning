#!/usr/bin/env python3


""" Contains transitions layer for DenseNet"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ Builds a transition layer"""
    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal')(X)
    X = K.layers.AveragePooling2D(2, strides=2)(X)
    return X, nb_filters
