#!/usr/bin/env python3


""" contains the def of Dataset class"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """Constructor"""
        data_train, data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                           split=['train', 'validation'],
                                           as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)
        data_train = data_train.map(self.tf_encode)

        def filter_max_length(x, y, max_length=max_len):
            """ filtering out sentences with length > max_length"""
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)
        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        data_train = data_train.shuffle(True).padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        data_valid = data_valid.map(self.tf_encode)
        data_train = data_train.filter(filter_max_length)
        self.data_valid = data_valid.shuffle(True).padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def filter_max_length(x, y, max_length=max_len):
        """ filtering out sentences with length > max_length"""
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)
    data_train = data_train.filter(filter_max_length)
    data_train = data_train.cache()
    data_train = data_train.shuffle(True).padded_batch(
        batch_size, padded_shapes=([None], [None]))
    self.data_train = data_train.prefetch(tf.data.AUTOTUNE)

    data_valid = data_valid.map(self.tf_encode)
    data_train = data_train.filter(filter_max_length)
    self.data_valid = data_valid.shuffle(True).padded_batch(
        batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """ Creates sub-word tokenizers for our dataset"""
        tokenize = tfds.features.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = tokenize((en.numpy()
                                 for pt, en in data), target_vocab_size=2**15)
        tokenizer_pt = tokenize((pt.numpy()
                                 for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        lang1 = [self.tokenizer_pt.vocab_size] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [self.tokenizer_pt.vocab_size + 1]
        lang2 = [self.tokenizer_en.vocab_size] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [self.tokenizer_en.vocab_size + 1]
        return lang1, lang2

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method"""
        result_pt, result_en = tf.py_function(
            self.encode, [
                pt, en], [
                tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
