#!/usr/bin/env python3


""" contains l2 reg cost"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ Returns: a tensor containing the cost
    of the network accounting for L2 regularization"""
    return cost + tf.losses.get_regularization_losses()
