#!/usr/bin/env python3


""" contains normalize funct"""
import numpy as np


def normalize(X, m, s):
    """ Returns: The normalized X matrix"""
    X -= m
    X /= s 
    return X
