#!/usr/bin/env python3
""" contains semantic_search function"""
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)


def semantic_search(corpus_path, sentence):
    """
    performs semantic search on a corpus of documents.
    """
    i = 0
    ar = [sentence]
    for filename in os.listdir(corpus_path):
        filename = os.path.join(corpus_path, filename)
        if filename.endswith(".md"):
            with open(filename) as f:
                # print("TITLE :" + filename)
                content = f.read()
                ar.append(content)
                # print("--------------------------------------------")
    embeddings = model(ar)
    corr = np.inner(embeddings, embeddings)
    argmax = np.argmax(corr[0, 1:])
    return ar[argmax + 1]
