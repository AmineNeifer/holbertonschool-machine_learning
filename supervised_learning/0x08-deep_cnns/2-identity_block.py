#!/usr/bin/env python3

""" function Identity block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ Builds an Identity block"""
    F11, F3, F12 = filters

    conv_1x1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        padding='same')(A_prev)
    conv_1x1 = K.layers.BatchNormalization(axis=3)(conv_1x1)
    conv_1x1 = K.layers.Activation('relu')(conv_1x1)
    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same')(conv_1x1)
    conv_3x3 = K.layers.BatchNormalization(axis=3)(conv_3x3)
    conv_3x3 = K.layers.Activation('relu')(conv_3x3)
    conv_1x1 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same')(conv_3x3)
    conv_1x1 = K.layers.BatchNormalization(axis=3)(conv_1x1)
    added = K.layers.Add()([A_prev, conv_1x1])
    output = K.layers.Activation('relu')(added)
    return output
