#!/usr/bin/env python3

""" contains func specificity"""
import numpy as np


def specificity(confusion):
    """ returns an array containing specificity"""
    actual = np.sum(confusion, axis=1)
    total = np.sum(confusion)
    actual_no = total - actual

    predicted = np.sum(confusion, axis=0)
    diagonal = np.diagonal(confusion)
    FP = predicted - diagonal

    return 1 - FP / actual_no