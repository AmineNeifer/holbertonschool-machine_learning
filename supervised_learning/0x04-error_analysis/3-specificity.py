#!/usr/bin/env python3

""" contains func specificity"""
import numpy as np


def specificity(confusion):
    """ returns an array containing specificity"""
    diag = np.diagonal(np.fliplr(confusion))
    sensitivity = diag / confusion.sum(axis=1) #TP / (TP + FN)
    precision = diag / confusion.sum(axis=0) #TP / (TP + FP)
    """ specificity ="""                         #TN / (TN + FP)
    
    return
