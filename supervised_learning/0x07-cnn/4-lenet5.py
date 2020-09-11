#!/usr/bin/env python3


""" lenet5 function implemented in tf v.1"""
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow.

    Arguments:
    x -- tf.placeholder of shape (m, 28, 28, 1)
        m is the number of images
    y -- tf.placeholder of shape (m, 10)

    Architecture:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes

    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    w = tf.contrib.layers.variance_scaling_initializer()

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv_layer = tf.layers.Conv2D(
        filters=6,
        kernel_size=(
            5,
            5),
        activation='relu',
        padding="same",
        kernel_initializer=w)(x)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    max_pool_layer = tf.layers.MaxPooling2D(
        pool_size=2, strides=(2, 2))(conv_layer)
    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv_layer1 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(
            5,
            5),
        activation='relu',
        padding="valid",
        kernel_initializer=w)(max_pool_layer)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    max_pool_layer1 = tf.layers.MaxPooling2D(
        pool_size=2, strides=(2, 2))(conv_layer1)
    # our array should be flatten to become 1D
    Flat = tf.layers.Flatten()(max_pool_layer1)
    # Fully connected layer with 120 nodes
    FC = tf.layers.Dense(
        units=128,
        activation='relu',
        kernel_initializer=w)(Flat)
    # Fully connected layer with 84 nodes
    FC1 = tf.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=w)(FC)
    # Fully connected layer with 10 nodes
    FC2 = tf.layers.Dense(units=10, kernel_initializer=w)(FC1)

    s = tf.nn.softmax(FC2)
    loss = tf.losses.softmax_cross_entropy(y, s)
    adam = tf.train.AdamOptimizer().minimize(loss)
    # accuracy
    equality = tf.equal(tf.argmax(s, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(equality, tf.float32))
    return s, adam, loss, acc
