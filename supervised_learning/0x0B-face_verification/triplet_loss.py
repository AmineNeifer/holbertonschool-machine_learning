#!/usr/bin/env python3

""" triplet loss layers makeing"""
import tensorflow as tf
from tensorflow import keras as K
import numpy as np


class TripletLoss(K.layers.Layer):
    """ class TripletLoss"""

    def __init__(self, alpha, **kwargs):
        """initialisation"""
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """ Return:  a tensor containing the triplet loss values"""
        anchor = inputs[0]
        positive = inputs[1]
        negative = inputs[2]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.maximum(pos_dist - neg_dist + self.alpha, 0.)
