#!/usr/bin/env python3

""" contains a function for deriv"""

def poly_derivative(poly):
    """ polynome derivative function"""
    if not poly:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(len(poly)):
        poly[i] *= i
    poly.pop(0)
    return poly
