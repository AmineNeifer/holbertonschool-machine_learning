#!/usr/bin/env python3

""" contains a function that computes sigma summation"""


def summation_i_squared(n):
    """ sigma summation"""
    if type(n) is not int:
        return None
    summ = 0
    for i in range(n):
        summ += (i+1) ** 2
    return summ
