#!/usr/bin/env python3


""" contains DeepNeuralNetwork definition"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork"""

    def __init__(self, nx, layers, activation="sig"):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activatin must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__activation = activation
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], nx) * \
                    np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * \
                    np.sqrt(2 / layers[i - 1])
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

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """ Calculates the forward propagation which is X.W + b (b = 0)"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            W = self.__weights["W" + str(i + 1)]
            A = self.__cache["A" + str(i)]
            b = self.__weights["b" + str(i + 1)]
            Z = np.matmul(W, A) + b
            if (i < self.__L - 1):
                if (self.__activation == 'sig'):
                    self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
                else:
                    self.__cache["A" + str(i + 1)] = 2 / \
                        (1 + np.exp(-2 * Z)) - 1
            else:
                t = np.exp(Z)
                self.__cache["A" + str(i + 1)] = t / \
                    np.sum(t, axis=0, keepdims=True)
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost"""
        array_sum = np.sum(Y * np.log(A))
        return - array_sum / Y.shape[1]

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        A, _ = self.forward_prop(X)
        tmp = np.amax(A, axis=0)
        return np.where(A == tmp, 1, 0), self.cost(Y, A)

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
            self.__weights["W" + str(i + 1)] = \
                self.__weights["W" + str(i + 1)] - (alpha * dW)
            self.__weights["b" + str(i + 1)] = \
                self.__weights["b" + str(i + 1)] - (alpha * db)
            if (self.__activation == 'sig'):
                dZ = np.matmul(
                    weighs["W" + str(i + 1)].T, dZ) * (A * (1 - A))
            else:
                dZ = np.matmul(
                    weighs["W" + str(i + 1)].T, dZ) * (1 - A ** 2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the neuron by updating the
        private attributes __W, __b, and __A"""
        it = []
        co = []
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose is True and graph is False) or \
                (verbose is False and graph is True):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        for i in range(0, iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (i == 0 or i % step == 0 or i == iterations):
                cost = self.cost(Y, A)
                it.append(i)
                co.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if verbose is True:
            cost = self.cost(Y, A)
            it.append(i)
            co.append(cost)
            print("Cost after {} iterations: {}".format(iterations, cost))
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
