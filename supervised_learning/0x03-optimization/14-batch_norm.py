#!/usr/bin/env python3

import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    gamma = tf.Variable(initial_value=1, dtype=tf.float32, trainable=True)
    beta = tf.Variable(initial_value=0, dtype=tf.float32, trainable=True)
    epsilon = tf.Variable(initial_value=1e-8)

    w = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')

    linear_model = tf.layers.Dense(units=n,
                                   kernel_initializer=w,
                                   activation=activation
                                   )
    y = linear_model(prev)
    mean, variance = tf.nn.moments(y, axes=0)
    
    return tf.nn.batch_normalization(y, mean, variance, beta, gamma, epsilon)