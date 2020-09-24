#!/usr/bin/env python3

""" momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using th egradient descent of momentum"""
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, grad
