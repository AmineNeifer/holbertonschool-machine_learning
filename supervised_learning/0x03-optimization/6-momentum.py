#!/usr/bin/env python3

""" momentum upgraded"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """ reates the training operation for a neural network in tensorflow"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
