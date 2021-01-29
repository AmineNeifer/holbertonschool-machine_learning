#!/usr/bin/env python3
""" Contains question_answer function"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
semantic_search = __import__('3-semantic_search').semantic_search
question_answer = __import__('0-qa').question_answer

tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def qa_bot(corpus_path):
    """answers questions from a reference text in command line like"""
    while (True):
        exit_keys = ["exit", "quit", "goodbye", "bye"]
        question = input("Q: ")
        print("A: ", end="")
        if question.strip().lower() in exit_keys:
            print("Goodbye")
            break
        reference = semantic_search(corpus_path, question)
        answer = question_answer(question, reference)
        if answer is None:
            print("Sorry, I do not understand your question.")
        else:
            print(answer)
