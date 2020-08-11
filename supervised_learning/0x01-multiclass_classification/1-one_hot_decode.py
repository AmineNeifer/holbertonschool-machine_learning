#!/usr/bin/env python3


""" contains one hot decoder function"""
import numpy as np


def one_hot_decode(one_hot):
    """ that converts a one-hot matrix into a vector of labels"""
    return np.argmax(one_hot, axis=0)
