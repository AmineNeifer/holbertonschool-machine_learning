#!/usr/bin/env python3

""" contains ngram_bleu function"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the ngram BLEU
    score for a sentence
    """
    c = len(sentence)
    r = np.array([len(r) for r in references])
    r = np.argmin(np.abs(r - c))
    r = len(references[r])
    bp = 1
    if r > c:
        bp = np.exp(1 - r / c)
    references, sentence = ngram_div(references, sentence, n)
    words = {}
    for word in sentence:
        for ref in references:
            if word in words:
                if words[word] < ref.count(word):
                    words.update({word: ref.count(word)})
            else:
                words.update({word: ref.count(word)})
    p = sum(words.values()) / len(sentence)
    print("bp: ", bp)
    print("p: ", p)
    return bp * p


def ngram_div(references, sentence, n):
    """
    divide sentences into ngrams
    """
    sent = []
    ref = []
    for i in range(len(sentence)):
        if (i + n > len(sentence)):
            break
        sent.append(sentence[i:i + n])
    sent = [' '.join(item) for item in sent]
    for j in range(len(references)):
        ref.append([])
        for i in range(len(references[j])):
            if (i + n > len(references[j])):
                break
            ref[j].append(references[j][i:i + n])
        ref[j] = [' '.join(item) for item in ref[j]]
    return ref, sent
