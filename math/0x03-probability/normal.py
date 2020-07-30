#!/usr/bin/env python3


""" contains class for normal distribution in probablity"""


class Normal:
    """ normal distibution class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        self.data = data
        if data is None:
            self.stddev = float(stddev)
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            n = len(data)
            m = self.mean
            self.stddev = (sum((x - m) ** 2 for x in data) / n) ** (1/2)

    def z_score(self, x):
        """ calculates the z-score of a given x-score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculates the value of PDF for a given x-value"""
        e = 2.7182818285
        pi = 3.1415926536
        o = self.stddev
        if x < 0:
            return 0
        return e ** ((-1/2) * self.z_score(x) ** 2) / (o * self.sqrt(2 * pi))

    def sqrt(self, x):
        """ returns square root of a number"""
        return x ** (1/2)
