#!/usr/bin/env python3


""" contains the Poisson distribution Class"""


class Poisson:
    """ Poisson distribution Class"""

    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
