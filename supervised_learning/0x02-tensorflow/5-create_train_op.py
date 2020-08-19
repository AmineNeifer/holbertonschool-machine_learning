#!/usr/bin/env python3


""" contains train funct"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """ train count funct"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
