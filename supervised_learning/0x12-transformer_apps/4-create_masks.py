#!/usr/bin/env python3

""" contains create_masks funct"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """creates all masks for training/validation"""
    batch_size, seq_len_out = target.shape
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]
    mask = tf.math.subtract(1,
                            tf.linalg.band_part(
                                tf.ones(
                                    (batch_size, 1, seq_len_out, seq_len_out)), -1, 0))
    decoder_mask = tf.cast(tf.math.equal(target, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]
    return encoder_mask, mask, decoder_mask
