#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    """ matrix multiplication of 2 2D matrices"""
    new = []
    for i in range(len(mat1)):
        new.append([])
        for j in range(len(mat2[0])):
            result = 0
            for c in range(len(mat1[0])):
                result += mat1[i][c] * mat2[c][j]
            new[i].append(result)
    return (new)