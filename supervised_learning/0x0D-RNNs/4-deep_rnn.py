#!/usr/bin/env python3


""" contains feed forward deep rnn function """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN:

    @rnn_cells is a list of RNNCell instances for the forward prop
        @length is the number of layers
    @X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        @t is the maximum number of time steps
        @m is the batch size
        @i is the dimensionality of the data
    @h_0 is the initial hidden state, given as an array shape (length, m, h)
        @h is the dimensionality of the hidden state
    Returns: H, Y
        @H is a numpy.ndarray containing all of the hidden states
        @Y is a numpy.ndarray containing all of the outputs
    """
    length = len(rnn_cells)
    h = h_0.shape[2]
    T, m, i = X.shape
    o = rnn_cells[-1].by.shape[1]
    H = np.zeros((T + 1, length, m, h))
    Y = np.zeros((T, m, o))
    H[0, ...] = h_0

    for t in range(1, T + 1):
        H[t, 0], Y[t - 1] = rnn_cells[0].forward(H[t - 1, 0], X[t - 1])
        for i in range(1, length):
            rnn_cell = rnn_cells[i]
            H[t, i], Y[t - 1] = rnn_cells[i].forward(H[t - 1, i], H[t, i - 1])
    return H, Y
