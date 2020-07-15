#!/usr/bin/env python3
""" contains a function that concatenates arrays"""


def cat_arrays(arr1, arr2):
    """ concatenate two arrays"""
    new = arr1.copy()
    for el in arr2:
        new.append(el)
    return(new)
