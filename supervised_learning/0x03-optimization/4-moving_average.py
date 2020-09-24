#!/usr/bin/env python3


""" moving average"""


def moving_average(data, beta):
    """ returns a list containing the moving averages of data"""
    value = 0
    listt = []
    for i in range(len(data)):
        value = (value * beta + (1 - beta) * data[i])
        listt.append(value / (1 - beta ** (i + 1)))
    return listt
