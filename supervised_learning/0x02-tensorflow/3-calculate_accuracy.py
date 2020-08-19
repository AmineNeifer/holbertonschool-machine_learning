#!/usr/bin/env python3


""" contains accuracy funct"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ accuracy count funct"""
    equality = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
