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
            split=['train', 'validate'],
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """ Creates sub-word tokenizers for our dataset"""
        tokenize = tfds.features.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = tokenize((en.numpy()
                                 for pt, en in data), target_vocab_size=2**15)
        tokenizer_pt = tokenize((pt.numpy()
                                 for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en
