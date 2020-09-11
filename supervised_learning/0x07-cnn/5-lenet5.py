#!/usr/bin/env python3

import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras

    Arguments:
    X -- K.Input of shape (m, 28, 28, 1) containing the input images

    Architecture:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes

    Returns:
    K.Model compiled to use Adam optimization and accuracy metrics
    """
    w = K.initializers.he_normal(seed=None)
    conv = K.layers.Conv2D(filters=6, kernel_size=5,
                           padding='same', activation='relu',
                           kernel_initializer=w)(X)
    pool = K.layers.MaxPooling2D(pool_size=2, strides=(2, 2))(conv)
    conv = K.layers.Conv2D(filters=16, kernel_size=5,
                           padding='valid', activation='relu',
                           kernel_initializer=w)(pool)
    pool = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv)
    flat = K.layers.Flatten()(pool)
    n = K.layers.Dense(units=120, activation='relu',
                       kernel_initializer=w)(flat)
    n = K.layers.Dense(units=84, activation='relu',
                       kernel_initializer=w)(n)
    n = K.layers.Dense(units=10, activation='softmax',
                       kernel_initializer=w)(n)
    model = K.models.Model(inputs=X, outputs=n)
    model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
