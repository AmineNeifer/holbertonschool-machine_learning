#!/usr/bin/env python3

""" contains func create_layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """ creates a layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')

    linear_model = tf.layers.Dense(units=n,
                                   name="layer",
                                   kernel_initializer=w,
                                   activation=activation
                                   )
    y = linear_model(prev)
    return y
