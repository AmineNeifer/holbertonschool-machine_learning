#!/usr/bin/env python3


""" contains DeepNeuralNetwork definition"""
import numpy as np


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], nx) * \
                    np.sqrt(2/nx)
            else:
                self.__weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i-1]) * \
                    np.sqrt(2/layers[i-1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation which is X.W + b (b = 0)"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.__weights["W" + str(i+1)]
            A = self.__cache["A" + str(i)]
            b = self.__weights["b" + str(i+1)]
            Z = np.matmul(W, A) + b
            self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(i + 1)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost"""
        c = - (np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        c /= Y.shape[1]
        return c

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron"""
        m = self.__cache["A0"].shape[1]
        length = self.__L
        A, _ = self.forward_prop(self.__cache["A0"])
        dZ = A - Y
        dW = np.matmul(dZ, self.__cache["A" + str(length)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        self.__cache["dZ" + str(length)] = dZ
        self.__weights["dW" + str(length)] = dW
        self.__weights["db" + str(length)] = db

        for i in range(self.__L - 1, 0, -1):
            A = self.__cache["A" + str(i)]
            dZ = np.matmul(
                self.__weights["W" + str(i+1)].T, dZ) * (A * (1 - A))
            dW = np.matmul(dZ, self.__cache["A" + str(i-1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W" + str(i)] = \
                self.__weights["W" + str(i)] - (alpha * dW)
            self.__weights["b" + str(i)] = \
                self.__weights["b" + str(i)] - (alpha * db)
