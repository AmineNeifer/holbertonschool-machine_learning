#!/usr/bin/env python3
""" function that adds two arrays element wise"""


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise"""
    new_l = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        new_l.append(arr1[i] + arr2[i])
    return(new_l)
