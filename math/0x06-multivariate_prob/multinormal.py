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
        self.cov = np.cov(data)
