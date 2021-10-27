#!/usr/bin/env python3

""" function Identity block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ Builds an Identity block"""
    F11, F3, F12 = filters

    conv_1x1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        padding='same', kernel_initializer="he_normal")(A_prev)
    conv_1x1 = K.layers.BatchNormalization()(conv_1x1)
    conv_1x1 = K.layers.ReLU()(conv_1x1)
    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same', kernel_initializer="he_normal")(conv_1x1)
    conv_3x3 = K.layers.BatchNormalization()(conv_3x3)
    conv_3x3 = K.layers.ReLU()(conv_3x3)
    conv_1x1 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        padding='same', kernel_initializer="he_normal")(conv_3x3)
    conv_1x1 = K.layers.BatchNormalization()(conv_1x1)

    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        padding='same', kernel_initializer="he_normal")(A_prev)
    shortcut = K.layers.BatchNormalization()(shortcut)
    added = K.layers.Add()([conv_1x1, shortcut])
    output = K.layers.ReLU()(added)
    return output
