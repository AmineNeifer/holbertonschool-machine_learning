#!/usr/bin/env python3

det = __import__('0-determinant').determinant
def minor(matrix):
    mat = [m[:] for m in matrix[:]]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            print(rest(mat, i, j))
    return (1)

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
