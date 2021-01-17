#!/usr/bin/env python3

""" contains MultiHeadAttention class"""
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """ class to perform multi head attention"""

    def __init__(self, dm, h):
        """class Constructor"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)  # for query matrix generation
        self.Wk = tf.keras.layers.Dense(dm)  # for key matrix generation
        self.Wv = tf.keras.layers.Dense(dm)  # for value matrix generation
        # for attention output generation
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """ returns output and weights"""
        batch_size = Q.shape[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # (batch_size, seq_len_q, d_model)
        output = self.linear(concat_attention)

        return output, attention_weights
