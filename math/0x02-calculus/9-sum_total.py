#!/usr/bin/env python3

""" contains a function that computes sigma summation"""


def summation_i_squared(n):
    """ sigma summation"""
    if not isinstance(n, int):
        return None
    if n == 1:
        return 1
    if n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
