#!/usr/bin/env python3
""" contains class MultiNormal"""
import numpy as np


class MultiNormal:
    """ Prob class MultiNormal"""

    def __init__(self, data):
        """ init functin for MultiNormal"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul((data - self.mean),
                             (data - self.mean).T) / (data.shape[1] - 1)
        self.data = data

    def pdf(self, x):
        """ pdf at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        m = self.mean
        E = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        denom = 1 / ((((2 * np.pi) ** (d)) * E) ** 0.5)
        nominator = np.exp(- 0.5 *
                           np.matmul(np.matmul((x - m).T, inv), (x - m)))
        return (nominator * denom)[0][0]
