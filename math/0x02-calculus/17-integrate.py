#!/usr/bin/env python3

""" file that contains an integrating function"""


def poly_integral(poly, C=0):
    """ integraling a polynom"""
    if not isinstance(poly, list) or not isinstance(C, int) or (not poly):
        return None
    if poly != [0]:
        poly_clone = poly[:]
    else:
        poly_clone = []
    poly_clone.insert(0, C)
    for i in range(1, len(poly_clone)):
        if (poly_clone[i] != 0):
            poly_clone[i] /= i
            if poly_clone[i] == int(poly_clone[i]):
                poly_clone[i] = int(poly_clone[i])
    return poly_clone
