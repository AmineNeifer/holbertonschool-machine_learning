#!/usr/bin/env python3


""" contains a function for deriv"""


def poly_derivative(poly):
    """ polynome derivative function"""
    if not poly:
        return None
    if len(poly) == 1:
        return [0]
    poly_clone = poly[:]
    for i in range(len(poly_clone)):
        poly_clone[i] *= i
    poly_clone.pop(0)
    return poly_clone
