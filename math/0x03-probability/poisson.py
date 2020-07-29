#!/usr/bin/env python3


""" contains the Poisson distribution Class"""


class Poisson:
    """ Poisson distribution Class"""
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
            self.lambtha = sum(data) / len(data)

    def factor(self, n):
        """ calculates the factorial of n"""
        fac = 1
        for i in range(1, n+1):
            fac *= i
        return fac

    def pmf(self, k):
        """ calculates the pmf for a given number of successes """
        e = 2.7182818285
        lamda = self.lambtha
        k = int(k)
        return lamda ** k * e ** (-lamda) / self.factor(k)
