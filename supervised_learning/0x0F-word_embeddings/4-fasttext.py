#!/usr/bin/env python3

""" contains fasttext_model function"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0):
    """ Creates and trains gensim fasttext model"""
    model = FastText(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=not cbow,
        iter=iterations,
        seed=seed)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs)
    return model
