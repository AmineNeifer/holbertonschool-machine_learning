#!/usr/bin/env python3


""" contains the def of Dataset class"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """Constructor"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

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
