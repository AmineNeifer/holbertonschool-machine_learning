#!/usr/bin/env python3


""" contains f1 score func"""
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ returns  F1 score of each class"""
    ppv = precision(confusion)
    tpr = sensitivity(confusion)
    return 2 * ppv * tpr / (ppv + tpr)
