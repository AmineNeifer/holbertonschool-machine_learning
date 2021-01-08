#!/usr/bin/env python3


"""
Create a class RNNEncoder that inherits from tensorflow.keras.layers.Layer
to encode for machine translation.
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """rnn encoder for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """ class constructor"""
        super(RNNEncoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """ initializes hidden state .-."""
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """ retruns outputs of the encoder and last hidden state"""
        emb = self.embedding(x)
        return self.gru(emb, initial)
