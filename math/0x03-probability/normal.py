#!/usr/bin/env python3


""" contains class for normal distribution in probablity"""


class Normal:
    """ normal distibution class"""
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            self.stddev = float(stddev)
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("stddev must be a positive value")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = self.stddeviation(data)

    def stddeviation(self, data):
        """ returns the standard deviation"""
        variance = sum((xi - self.mean) ** 2 for xi in data) / len(data)
        return variance ** (1/2)
