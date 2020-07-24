#!/usr/bin/env python3

""" contains a function that computes sigma summation"""


def summation_i_squared(n):
    """ sigma summation"""
    if type(n) is not int:
        return None

    if n == 1:
        return 1
    if n < 0:
        return n ** 2 + summation_i_squared(n+1)
    else:
        return n ** 2 + summation_i_squared(n-1)
