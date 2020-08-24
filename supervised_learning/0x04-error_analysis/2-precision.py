#!/usr/bin/env python3

""" contains func precision"""
import numpy as np


def precision(confusion):
    """ returns an array containing precision"""
    diag = np.diagonal(confusion)
    return (diag / confusion.sum(axis=0))
