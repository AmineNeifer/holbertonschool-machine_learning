#!/usr/bin/env python3

"""
Contains class GRUCell that represents a cell of a GRU RNN
"""
import numpy as np


class GRUCell:
    """represents a cell of a GRU RNN"""

    def __init__(self, i, h, o):
        """
        Class constructor:

        @i is the dimensionality of the data
        @h is the dimensionality of the hidden state
        @o is the dimensionality of the outputs
        - Creates the public instance attributes that represent
        the weights and biases of the cell
            @Wz and @bz are for the update gate
            @Wr and @br are for the reset gate
            @Wh and @bh are for the intermediate hidden state
            @Wy and @by are for the output
        """
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """sigmoid fucntion"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
            @x_t is a np.ndarray (m, i) contains the data input for the cell
                m is the batche size for the data
            @h_prev is a np.ndarray (m, h) contains the previous hidden state

        Returns: h_next, y
            @h_next is the next hidden state
            @y is the output of the cell
        """
        comb = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(comb.dot(self.Wz) + self.bz)
        r = self.sigmoid(comb.dot(self.Wr) + self.br)
        comb = np.concatenate((h_prev * r, x_t), axis=1)
        h = np.tanh(comb.dot(self.Wh) + self.bh)
        h_t = z * h + (1 - z) * h_prev
        y = self.softmax(h_t.dot(self.Wy) + self.by)
        return h_t, y
