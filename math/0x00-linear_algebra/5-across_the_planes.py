#!/usr/bin/env python3


def add_matrices2D(mat1, mat2):
    """ add two 2D matrices"""
    if (len(mat1) != len(mat2)) or (len(mat1[0]) != len(mat2[0])):
        return None
    new = []
    for el in mat1:
        new.append([])
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            new[i].append(mat1[i][j] + mat2[i][j])
    return new
