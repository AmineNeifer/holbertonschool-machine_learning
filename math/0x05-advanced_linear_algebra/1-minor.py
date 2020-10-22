#!/usr/bin/env python3


""" doing the minor of a matrix"""
det = __import__('0-determinant').determinant


def minor(matrix):
    """ takes a matrix and returns it's minor"""
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    elif len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return [[1]]

    mat = [m[:] for m in matrix[:]]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = det(rest(matrix, i, j))
    return mat


def rest(m, i, j):
    """
    returns the matrix we are going to do the
    deteminant for after getting the pivot
    """
    new = [part[:] for part in m[:]]
    for part in new:
        part.pop(j)
    new.pop(i)
    return new
