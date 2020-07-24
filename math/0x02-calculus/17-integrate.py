#!/usr/bin/env python3

""" file that contains an integrating function"""


def poly_integral(poly, C=0):
    """ integraling a polynom"""
    if not poly or  type(C) is not int:
        return None
    poly_clone = poly[:]
    poly_clone.insert(0, C)
    for i in range(1, len(poly_clone)):
        if (poly_clone[i] != 0):
            poly_clone[i] /= i
    return poly_clone
