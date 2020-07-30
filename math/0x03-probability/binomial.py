#!/usr/bin/env python3


""" contains class for binomial distribution in probablity"""


class Binomial:
    """ binomial distibution class"""

    def __init__(self, data=None, n=1, p=0.5):
        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.n = len(data)
            mean = sum(data)/len(data)
            var = [(i - mean)**2 for i in data]
            cat = sum(var)/len(data)
            self.p = 1 - (cat/mean)
            self.n = int(round(mean/self.p))
            self.p = mean/self.n
