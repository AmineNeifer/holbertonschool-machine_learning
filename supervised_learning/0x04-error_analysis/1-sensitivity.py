#!/usr/bin/env python3

""" contains func sensitivity"""
import numpy as np


def sensitivity(confusion):
    """ returns an array containing sensitivity"""
    diag = np.diagonal(confusion)
    return (diag / confusion.sum(axis=1))
