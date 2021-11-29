#!/usr/bin/env python3


""" function Identity block"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ Builds ResNet-50 architecture"""
    input_layer = K.Input([224, 224, 3])
    w = 'he_normal'

    conv_7x7 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2,
        kernel_initializer=w)(input_layer)
    conv_7x7 = K.layers.BatchNormalization()(conv_7x7)
    conv_7x7 = K.layers.ReLU()(conv_7x7)

    x = K.layers.MaxPool2D(
        (3, 3),
        strides=2, padding='same')(conv_7x7)

    x = projection_block(x, (64, 64, 256), s=1)
    for i in range(2):
        x = identity_block(x, (64, 64, 256))

    x = projection_block(x, (128, 128, 512), s=2)
    for i in range(3):
        x = identity_block(x, (128, 128, 512))

    x = projection_block(x, (256, 256, 1024), s=2)
    for i in range(5):
        x = identity_block(x, (256, 256, 1024))

    x = projection_block(x, (512, 512, 2048), s=2)
    for i in range(2):
        x = identity_block(x, (512, 512, 2048))
    x = K.layers.AveragePooling2D((7, 7), strides=2)(x)
    x = K.layers.Dense(units=1000, activation='softmax')(x)
    model = K.models.Model(input_layer, x)
    return model
