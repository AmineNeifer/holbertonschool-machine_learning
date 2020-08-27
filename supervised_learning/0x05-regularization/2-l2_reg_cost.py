#!/usr/bin/env python3


""" contains l2 reg cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """ Returns: a tensor containing the cost
    of the network accounting for L2 regularization"""
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01
    loss = cost + reg_constant * sum(reg_losses)
    return loss
