#!/usr/bin/env python3


"""adam optimization with tensorflow"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ adam opt"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
