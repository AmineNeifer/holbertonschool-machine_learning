#!/usr/bin/env python3
""" contains a function that returns
sum, diff, prod and div between 2 matrices"""


def np_elementwise(mat1, mat2):
    """ returns elementwise sum, diff, prod and div"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
