#!/usr/bin/env python3

""" contains word2vec_model function"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """ Creates and trains gensim word2vec model"""
    sg = not cbow
    model = Word2Vec(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        iter=iterations,
        seed=seed,
        workers=workers)
    # model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs)
    return model
