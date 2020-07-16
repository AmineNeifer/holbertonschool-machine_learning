#!/usr/bin/env python3
""" contains a function that return the shape of a matrix"""


def matrix_shape(matrix):
    """ calculates the shape of matrix"""
    shape = []
    while True:
        try:
            shape.append(len(matrix))
            matrix = matrix[0]
        except TypeError:
            return shape
