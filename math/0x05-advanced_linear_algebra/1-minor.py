#!/usr/bin/env python3


""" doing the minor of a matrix"""


def minor(matrix):
    """ takes a matrix and returns it's minor"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    elif len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    for item in matrix:
        if not isinstance(item, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    mat = [m[:] for m in matrix[:]]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = det(rest(matrix, i, j))
    return mat


def det(matrix):
    """ return the determinant of matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    for item in matrix:
        if not isinstance(item, list):
            raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    elif len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    mat = [a[:] for a in matrix[:]]
    return det_helper(mat)


def det_helper(mat):
    """ helper for deteminant, is the recursive funct"""
    if len(mat) == 1:
        return mat[0][0]
    if len(mat) == 2:
        return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]
    det = 0
    a = -1
    for i in range(len(mat)):
        a *= -1
        m = [part[:] for part in mat[:]]
        r = rest(m, 0, i)
        det += a * m[0][i] * det_helper(r)
    return det


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
