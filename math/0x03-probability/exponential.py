#!/usr/bin/env python3


""" contains the class Exponential"""


class Exponential:
    """ Exponential class in probability"""

    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """ returns pdf of exponential"""
        lamda = self.lambtha
        e = 2.7182818285
        if x < 0:
            return 0
        return lamda * e ** (-lamda * x)

    def cdf(self, x):
        """ returns cdf of exponential"""
        lamda = self.lambtha
        e = 2.7182818285
        if x < 0:
            return 0
        return 1 - e ** (-lamda * x)
