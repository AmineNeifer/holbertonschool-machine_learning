#!/usr/bin/env python3


""" contains loss funct"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """ loss count funct"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
