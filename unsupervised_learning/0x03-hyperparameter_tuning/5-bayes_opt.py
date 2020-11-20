#!/usr/bin/env python3


""" contains Bayesian Optimization"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        return self.X_s[np.argmax(ei)], ei

    def optimize(self, iterations=100):
        """Optimizes the black-box function"""
        l_ = []

        for _ in range(iterations):
            x, _ = self.acquisition()
            if x in l_:
                break

            y = self.f(x)
            l_.append(x)
            self.gp.update(x, y)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        self.gp.X = np.delete(self.gp.X, -1).reshape((-1, 1))

        return self.gp.X[index], self.gp.Y[index]
