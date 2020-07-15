#!/usr/bin/env python3
""" matrix function that computes transpose"""


def matrix_transpose(matrix):
    """ returns transpose of a matrix"""
    new = []
    for el in matrix[0]:
        new.append([])
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            new[i].append(matrix[j][i])
    return(new)
