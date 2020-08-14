#!/usr/bin/env python3


""" contains class Neuron definition"""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron"""
        m = X.shape[1]
        dZ = A - Y
        dW = np.sum(X * dZ, axis=1)
        db = np.sum(dZ)
        dW /= m
        db /= m
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the neuron by updating the
        private attributes __W, __b, and __A"""
        it = []
        co = []
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose is True and graph is False) or \
                (verbose is False and graph is True):
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if verbose is True:
            for i in range(0, iterations + 1):
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A, alpha)
                cost = self.cost(Y, self.__A)
                it.append(i)
                co.append(cost)
                if (i == 0 or i % step == 0 or i == iterations):
                    print("Cost after {} iterations: {}".format(i, cost))
        else:
            for i in range(0, iterations):
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A, alpha)
                cost = self.cost(Y, self.__A)
        if graph is True:
            plt.plot(it, co)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
