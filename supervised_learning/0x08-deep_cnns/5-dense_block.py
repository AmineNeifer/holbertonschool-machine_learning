#!/usr/bin/env python3


""" Contains Dense_block funct"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ dense_block (Bottleneck Layer) implementation"""
    for i in range(layers):
        last = X
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(filters=nb_filters, kernel_size=1, padding='same')(X)
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(filters=nb_filters, kernel_size=3, padding='same')(X)
        x_list = [last,X]
        X = K.layers.Concatenate(axis=-1)(x_list)
        nb_filters += growth_rate
    return X, nb_filters

    #x = K.layers.BatchNormalization()(X)
    #x = K.layers.Activation('relu')(x)
    #x = K.layers.Conv2D(filters=nb_filters,kernel_size=1, padding='same')

