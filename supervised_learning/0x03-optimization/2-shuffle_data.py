#!/usr/bin/env python3


""" contains shuffle_data funct"""
import numpy as np


def shuffle_data(X, Y):
    """ Returns: X & Y shuffled"""
    return np.random.permutation(X), np.random.permutation(Y)