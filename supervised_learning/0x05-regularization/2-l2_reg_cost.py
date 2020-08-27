#!/usr/bin/env python3


""" contains l2 reg cost"""
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost):
    """ Returns: a tensor containing the cost
    of the network accounting for L2 regularization"""
    return tf.Variable(cost)
