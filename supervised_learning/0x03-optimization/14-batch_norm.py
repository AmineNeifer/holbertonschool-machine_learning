#!/usr/bin/env python3

""" haw comments """
import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    """ comments marokhra"""
    gamma = tf.Variable(1, name="gamma", trainable=True, dtype=tf.float32)
    beta = tf.Variable(0, name="beta", trainable=True, dtype=tf.float32)
    epsilon = 1e-8
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(n, kernel_initializer=w)
    z = dense(prev)
    m, v = tf.nn.moments(z, axes=0)
    z_norm = tf.nn.batch_normalization(z, m, v, beta, gamma, epsilon)
    a = activation(z_norm)
    return a