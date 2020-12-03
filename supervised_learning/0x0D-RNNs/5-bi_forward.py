#!/usr/bin/env python3

"""
Contains class BidirectionalCell that represents a bidirectional cell RNN
"""
import numpy as np


class BidirectionalCell:
    def __init__(self, i, h, o):
        """
        Class constructor:

        @i is the dimensionality of the data
        @h is the dimensionality of the hidden state
        @o is the dimensionality of the outputs
        - Creates the public instance attributes
            @Whf and @bhf are for the hidden states in the forward direction
            @Whb and @bhb are for the hidden states in the backward direction
            @Wy and @by are for the outputs
        """
        self.Whf = np.random.normal(size=(h + i, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ feed forward algo for RNN"""
        print("h_prev = " + str(h_prev.shape))
        print("x_t = " + str(x_t.shape))
        conc = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(conc.dot(self.Whf) + self.bhf)
        return h_next
