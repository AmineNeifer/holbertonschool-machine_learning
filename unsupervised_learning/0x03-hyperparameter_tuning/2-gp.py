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

    def predict(self, X_s):
        """ Predicts the mean and standard deviation of points in a GP"""
        X = self.X
        Y = self.Y
        kernel = self.kernel
        K = kernel(X, X)
        K_s = kernel(X, X_s)
        K_ss = kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(Y).reshape((-1,))
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu_s, np.diag(cov_s)

    def update(self, X_new, Y_new):
        """ Updates a Gaussian Process"""
        self.X = np.append(self.X, X_new).reshape((-1, 1))
        self.Y = np.append(self.Y, Y_new).reshape((-1, 1))
        self.K = self.kernel(self.X, self.X)
