#!/usr/bin/env python3


def add_matrices(mat1, mat2):
    """ function that adds two matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    
    if (is_one_dimenstion(mat1)):
        new = mat1[:]
    else:
        new = list(map(list, mat1))
    return added(new, mat2)

def added(new, mat2):
    if type(new[0]) is list:
        return added()
    

def num_of_dimentions(mat1):
    """ function returns dim of a mat"""
    if (is_one_dimenstion(mat1)):
        new = mat1[:]
    else:
        new = list(map(list, mat1))
    dim = 0
    while True:
        dim += 1
        if type(new[0]) is not int:
            new = new[0]
        else:
            break
    return dim

def is_one_dimenstion(mat):
    """ function returns True if matrix is one-dim, False otherwise"""
    for i in mat:
        if type(i) is list:
            return False
    return True
def matrix_shape(matrix):
    """ calculates the shape of matrix"""
    shape = []
    while True:
        try:
            shape.append(len(matrix))
            matrix = matrix[0]
        except TypeError:
            return shape