#!/usr/bin/env python3


""" contains DeepNeuralNetwork definition"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        weighs = self.__weights.copy()
        m = self.__cache["A0"].shape[1]
        length = self.__L
        A, _ = self.forward_prop(self.__cache["A0"])
        dZ = A - Y

        for i in range(self.__L - 1, -1, -1):
            A = self.__cache["A" + str(i)]

            dW = np.matmul(dZ, A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W" + str(i+1)] = \
                self.__weights["W" + str(i+1)] - (alpha * dW)
            self.__weights["b" + str(i+1)] = \
                self.__weights["b" + str(i+1)] - (alpha * db)
            dZ = np.matmul(
                weighs["W" + str(i+1)].T, dZ) * (A * (1 - A))

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
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if verbose is True:
            for i in range(0, iterations + 1):
                A, cache = self.forward_prop(X)
                self.gradient_descent(Y, cache, alpha)
                cost = self.cost(Y, A)
                it.append(i)
                co.append(cost)
                if (i == 0 or i % step == 0):
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(it, co)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format"""
        if not(filename.endswith(".pkl")):
            filename = filename + ".pkl"
        with open(filename, 'wb') as fileObject:
            pickle.dump(self, fileObject)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as fileObject:
                saved = pickle.load(fileObject)
            return saved
        except FileNotFoundError:
            return None
