#!/usr/bin/env python3


""" contains l2 reg cost funct"""
import numpy as np


def l2_reg_cost(cost, lambtha, weight, L, m):
    """ Returns: the cost of the network accounting for L2 regularization"""
    SUM = 0
    for i in range(L):
        SUM += np.linalg.norm(weight['W' + str(i + 1)])
    return cost + SUM * lambtha * 0.5 / m
