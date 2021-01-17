#!/usr/bin/env python3


"""contains DecoderBlock class"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ class to create an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Class Constructor"""
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def point_wise_feed_forward_network(self):
        """ puts self.dense_hidden and self.dense_output in one"""
        return tf.keras.Sequential([
            self.dense_hidden,  # (batch_size, seq_len, dff)
            self.dense_output  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        """
        Returns:
            a tensor of shape (batch, target_seq_len, dm)
            containing the block’s output
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(
            out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn = self.point_wise_feed_forward_network()
        ffn_output = ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3
