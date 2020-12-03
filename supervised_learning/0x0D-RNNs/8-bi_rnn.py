#!/usr/bin/env python3


""" contains feed forward bi rnn function """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bi RNN:

    @bi_cell is an instance of BidirectionalCell that will be used for the forward prop
    @X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        @t is the maximum number of time steps
        @m is the batch size
        @i is the dimensionality of the data
    @h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        @h is the dimensionality of the hidden state
    @h_t is the initial hidden state in the backward direction
    Returns: H, Y
        @H is a numpy.ndarray containing all of the hidden states
        @Y is a numpy.ndarray containing all of the outputs
    """
    T, m, i = X.shape
    o = bi_cell.by.shape[1]
    h = h_0.shape[1]
    Hf = np.zeros((T + 1, m, h))
    Hb = np.zeros((T + 1, m, h))
    Y = np.zeros((T, m, o))
    times = T
    Hf[0] = h_0
    Hb[-1] = h_t
    for t in range(1, T+1):
        Hf[t]= bi_cell.forward(Hf[t - 1], X[t - 1])
    for t in reversed(range(T)):
        Hb[t]= bi_cell.backward(Hb[t + 1], X[t])

    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
