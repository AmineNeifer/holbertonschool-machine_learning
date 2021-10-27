#!/usr/bin/env python3


""" Contains Dense_block funct"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ Builds the densenet 121 architecture"""
    input_layer = K.Input([224, 224, 3])
    w = 'he_normal'
    nb_filters = 64

    x = K.layers.BatchNormalization()(input_layer)
    x = K.layers.ReLU()(x)
    x = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=w)(x)
    x = K.layers.MaxPool2D((3, 3), strides=2, padding='same')(x)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)
    x = K.layers.AveragePooling2D(x.shape[1:3])(x)
    x = K.layers.Dense(units=1000, activation='softmax')(x)

    model = K.models.Model(input_layer, x)
    return model
