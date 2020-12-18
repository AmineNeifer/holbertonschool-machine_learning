#!/usr/bin/env python3

""" import sklearn and contains tf_idf  algo"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """creates a tf_idf embedding matrix"""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer.get_feature_names()
