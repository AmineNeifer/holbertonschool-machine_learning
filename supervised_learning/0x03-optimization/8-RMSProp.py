#!/usr/bin/env python3


""" RMSPROP USING TENSORFLOW"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ RMSProp"""
    return tf.train.RMSPropOptimizer(
        alpha, beta2, epsilon=epsilon).minimize(loss)
