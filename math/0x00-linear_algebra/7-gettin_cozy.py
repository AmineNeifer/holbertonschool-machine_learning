#!/usr/bin/env python3
""" contains a function that concatenates two matrices using axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concat two 2D matrices"""
    new = []
    for el in mat1:
        new.append(el[:])
    if axis == 0:
        for el in mat2:
            new.append(el)
    elif axis == 1:
        for i in range(len(mat2)):
            for el in (mat2[i]):
                new[i] += [el]
    return new
