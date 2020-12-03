#!/usr/bin/env python3

"""
Contains class RNNCell that represents a cell of a simple RNN
"""
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """
        Class constructor:

        @i is the dimensionality of the data
        @h is the dimensionality of the hidden state
        @o is the dimensionality of the outputs
        - Creates the public instance attributes @Wh, @Wy, @bh, @by
            @Wh and @bh are for the concatenated hidden state and input data
            @Wy and @by are for the output
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ feed forward algo for RNN"""
        conc = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(conc.dot(self.Wh) + self.bh)
        y = self.softmax(h_next.dot(self.Wy) + self.by)
        return h_next, y
