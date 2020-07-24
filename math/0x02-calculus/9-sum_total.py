#!/usr/bin/env python3

""" contains a function that computes sigma summation"""


def summation_i_squared(n):
    """ sigma summation"""
    if type(n) is not int:
        return None

    if n in (0, 1):
        return n

    return n ** 2 + summation_i_squared(n-1)
