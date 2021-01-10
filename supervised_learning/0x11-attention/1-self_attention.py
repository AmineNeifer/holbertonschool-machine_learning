#!/usr/bin/env python3


"""
Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer
to calculate attention for machine translation based on Dzmitry et Al. paper.
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """rnn encoder for machine translation"""

    def __init__(self, units):
        """ class constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ returns context vector and weights"""
        input_prep = self.W(tf.expand_dims(s_prev, 1))  # w
        score = self.U(hidden_states)  # u
        somme = self.V(tf.tanh(input_prep + score))  # e
        weights = tf.nn.softmax(somme, 1)  # s
        context = weights * hidden_states
        return tf.math.reduce_sum(context, 1), weights
