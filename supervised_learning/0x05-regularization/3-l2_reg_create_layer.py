#!/usr/bin/env python3


""" Contains l2_reg_create_layer funct"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ create a layer with l2 regularization"""
    reg = tf.keras.regularizers.l2(l=lambtha)
    w = tf.keras.initializers.VarianceScaling(scale=2.0, mode=('fan_avg'))
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=w,
        kernel_regularizer=reg,
        activation=activation)
    return layer(prev)
