#!/usr/bin/env python3


""" contains class Neuron definition"""
import numpy as np


class Neuron:
    """ Class Neuron"""

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation which is X.W + b (b = 0)"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost"""
        c = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        c /= Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost
