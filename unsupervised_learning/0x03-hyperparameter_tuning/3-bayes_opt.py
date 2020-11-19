#!/usr/bin/env python3


""" contains Bayesian Optimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D GP"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ Class constructor"""
        bo = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.X_s = np.linspace(bo[0], bo[1], ac_samples).reshape((-1, 1))
        self.minimize = minimize
