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
