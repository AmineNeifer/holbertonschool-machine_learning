#!/usr/bin/env python3


""" Contains dropout_create_layer funct"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ create a layer with dropout regularization"""
    reg = tf.layers.Dropout(1-keep_prob)
    w = tf.keras.initializers.VarianceScaling(scale=2.0, mode=('fan_avg'))
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=w,
        kernel_regularizer=reg,
        activation=activation)
    return layer(prev)
