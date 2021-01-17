#!/usr/bin/env python3

""" contains EncoderBlock class"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ Create an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ class constructor"""
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def point_wise_feed_forward_network(self):
        """ puts self.dense_hidden and self.dense_output in one"""
        return tf.keras.Sequential([
            self.dense_hidden,  # (batch_size, seq_len, dff)
            self.dense_output  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, training, mask=None):
        """
        Returns:
            a tensor of shape (batch, input_seq_len, dm)
            containing the block’s output
        """
        attn_output, _ = self.mha(
            x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn = self.point_wise_feed_forward_network()
        ffn_output = ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
