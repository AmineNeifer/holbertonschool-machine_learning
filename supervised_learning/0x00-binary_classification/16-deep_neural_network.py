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
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        for element in layers:
            if element <= 0:
                raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            if i == 0:
                self.weights['W' + str(i + 1)
                             ] = np.random.randn(layers[i], nx) * np.sqrt(2/nx)
            else:
                self.weights['W' + str(i + 1)] = \
                    np.random.randn(layers[i], layers[i-1]) * \
                    np.sqrt(2/layers[i-1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
