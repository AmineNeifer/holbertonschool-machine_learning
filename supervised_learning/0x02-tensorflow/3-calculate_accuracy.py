#!/usr/bin/env python3


""" contains accuracy funct"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """ accuracy count funct"""
    equality = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
