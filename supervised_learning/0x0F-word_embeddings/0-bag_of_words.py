#!/usr/bin/env python3

""" import sklearn and contains BOW algo"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix"""
    vectorizer = CountVectorizer()
    if vocab is None:
        vocab = sentences
    bag_of_words = vectorizer.fit(vocab)
    bag_of_words = vectorizer.transform(sentences)
    return bag_of_words.toarray(), vectorizer.get_feature_names()
