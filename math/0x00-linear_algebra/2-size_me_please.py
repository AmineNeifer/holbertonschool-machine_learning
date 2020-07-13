#!/usr/bin/env python3


def matrix_shape(matrix):
    """ calculates the shape of matrix"""
    try:
        return([len(matrix), len(matrix[0]), len(matrix[0][0])])
    except TypeError:
        pass
    try:
        return([len(matrix), len(matrix[0])])
    except TypeError:
        return([len(matrix)])
