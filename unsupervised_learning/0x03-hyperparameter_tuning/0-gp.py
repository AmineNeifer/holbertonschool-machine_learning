#!/usr/bin/env python3


""" Contains Gaussian Process Class def"""
import numpy as np


class GaussianProcess:
    """ Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Isotropic squared exponential kernel."""
        sigma_f, l_ = self.sigma_f, self.l
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l_**2 * sqdist)
