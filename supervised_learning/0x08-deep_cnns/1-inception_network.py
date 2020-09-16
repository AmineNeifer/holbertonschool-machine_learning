#!/usr/bin/env python3

""" function Inception (GoogleNet)"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network
    """

    input_layer = K.layers.Input(shape=(224, 224, 3))
    x = K.layers.Conv2D(
        64, (7, 7), padding='same', strides=(
            2, 2), activation='relu')(input_layer)
    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = K.layers.Conv2D(
        64, (1, 1), padding='valid', strides=(
            1, 1), activation='relu')(x)
    x = K.layers.Conv2D(
        192, (3, 3), padding='same', strides=(
            1, 1), activation='relu')(x)
    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    filters = (64, 96, 128, 16, 32, 32)
    x = inception_block(x,
                        filters)

    filters = (128, 128, 192, 32, 96, 64)
    x = inception_block(x,
                        filters)

    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    filters = (192, 96, 208, 16, 48, 64)
    x = inception_block(x,
                        filters)

    filters = (160, 112, 224, 24, 64, 64)
    x = inception_block(x,
                        filters)

    filters = (128, 128, 256, 24, 64, 64)
    x = inception_block(x,
                        filters)

    filters = (112, 144, 288, 32, 64, 64)
    x = inception_block(x,
                        filters)

    filters = (256, 160, 320, 32, 128, 128)
    x = inception_block(x,
                        filters)

    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    filters = (256, 160, 320, 32, 128, 128)
    x = inception_block(x,
                        filters)

    filters = (384, 192, 384, 48, 128, 128)
    x = inception_block(x,
                        filters)
    x = K.layers.AveragePooling2D((7, 7), strides=1)(x)
    x = K.layers.Dropout(0.4)(x)

    x = K.layers.Dense(units=1000, activation='softmax')(x)

    model = K.models.Model(input_layer, x)
    return model
