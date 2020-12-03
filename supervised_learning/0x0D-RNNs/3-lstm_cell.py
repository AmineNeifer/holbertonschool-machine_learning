#!/usr/bin/env python3

"""
Contains class LSTMCell that represents a cell of a GRU RNN
"""
import numpy as np


class LSTMCell:
    def __init__(self, i, h, o):
        """
        Class constructor:

        @i is the dimensionality of the data
        @h is the dimensionality of the hidden state
        @o is the dimensionality of the outputs
        - Creates the public instance attributes that represent the weights and biases of the cell
            @Wf and @bf are for the update gate
            @Wu and @bu are for the reset gate
            @Wc and @bc are for the intermediate hidden state
            @Wo and @bo are for the output
            @Wy and @by are for the output
        """
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
    def sigmoid(self, x):
        """sigmoid fucntion"""
        return 1 / (1 + np.exp(-x))
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step
            @x_t is a np.ndarray (m, i) contains the data input for the cell
                m is the batche size for the data
            @h_prev is a np.ndarray(m, h) contains the prev hidden state
            @h_prev is a np.ndarray (m, h) contains the previous hidden state
            
        Returns: h_next, y
            @h_next is the next hidden state
            @h_prev is the next cell state
            @y is the output of the cell
        """
        comb = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(comb.dot(self.Wf) + self.bf)
        u = self.sigmoid(comb.dot(self.Wu) + self.bu)
        c = np.tanh(comb.dot(self.Wc) + self.bc)
        c = f * c_prev + u * c
        o_t = self.sigmoid(comb.dot(self.Wo) + self.bo)
        h_t = o_t * np.tanh(c)
        y = self.softmax(h_t.dot(self.Wy) + self.by)
        return h_t, c, y
