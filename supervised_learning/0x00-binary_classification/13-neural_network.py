#!/usr/bin/env python3


""" contains class NeuralNetwork definition"""
import numpy as np


class NeuralNetwork:
    """ Class NeuralNetwork"""

    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def W2(self):
        return self.__W2

    @property
    def b1(self):
        return self.__b1

    @property
    def b2(self):
        return self.__b2

    @property
    def A1(self):
        return self.__A1

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation which is X.W + b (b = 0)"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost"""
        c = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        c /= Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        return np.where(A2 >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron"""
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__b2 = self.__b2 - (alpha * db2)
