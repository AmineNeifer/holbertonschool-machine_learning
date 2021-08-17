#!/usr/bin/env python3

""" haw comments """
import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    """ comments marokhra"""
    dense = tf.layers.Dense(n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
    z = dense(prev)
    gamma = tf.Variable(1, dtype=tf.float32, trainable=True)
    beta = tf.Variable(0, dtype=tf.float32, trainable=True)
    m, v = tf.nn.moments(z, axes=0)
    z_norm = tf.nn.batch_normalization(z, m, v, beta, gamma, 1e-8)
    return activation(z_norm)