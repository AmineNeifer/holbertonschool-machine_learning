#!/usr/bin/env python3


"""
Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer
to decode for machine translation.
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """rnn decoder for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """ class constructor"""
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")
        self.F = tf.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        returns:
        y: output words as a one-hot vector in targert vocab
        s: new decoder hidden state
        """
        batch = x.shape[0]
        x = self.embedding(x)

        selfattention = SelfAttention(self.units)
        context, _ = selfattention(s_prev, hidden_states)

        concat = tf.concat([tf.expand_dims(context, 1), x], -1)
        outputs, hidden = self.gru(concat)
        output = self.F(outputs)

        return tf.reshape(output, (batch, -1)), hidden
