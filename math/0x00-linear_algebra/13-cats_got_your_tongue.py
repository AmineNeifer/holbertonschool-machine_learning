#!/usr/bin/env python3
""" contains a function that oncatenates two matrices"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concatenate two matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
