#!/usr/bin/env python3

""" triplet loss layers makeing"""
import tensorflow as tf
from tensorflow import keras as K


class TripletLoss(K.layers.Layer):
    """ class TripletLoss"""

    def __init__(self, alpha, **kwargs):
        """initialisation"""
        super(TripletLoss, self).__init__()
        self.alpha = alpha
