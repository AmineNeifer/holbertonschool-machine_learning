#!/usr/bin/env python3

import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    dense = tf.layers.Dense(n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), name='dense')
    z = dense(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma', trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta', trainable=True)
    m, v = tf.nn.moments(z, axes=0)
    z_norm = tf.nn.batch_normalization(z, m, v, beta, gamma, 1e-8)
    a = activation(z_norm)
    return a