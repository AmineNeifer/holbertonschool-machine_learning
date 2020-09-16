#!/usr/bin/env python3

""" function Inception (GoogleNet)"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ Builds an inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv_1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=1,
        activation='relu')(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3R,
        padding='same',
        kernel_size=1,
        activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        padding='same',
        kernel_size=3,
        activation='relu')(conv_3x3)

    conv_5x5 = K.layers.Conv2D(
        filters=F5R,
        padding='same',
        kernel_size=1,
        activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(
        filters=F5,
        padding='same',
        kernel_size=5,
        activation='relu')(conv_5x5)

    pool_proj = K.layers.MaxPool2D(
        pool_size=3, strides=1, padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(
        filters=FPP,
        padding='same',
        kernel_size=1,
        activation='relu')(pool_proj)

    output = K.layers.concatenate(
        [conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
    return output
