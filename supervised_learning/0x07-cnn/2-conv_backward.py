#!/usr/bin/env python3


""" contains the conv back prop funct"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    Arguments:
    dZ -- is a numpy.ndarray the partial derivatives with
    respect to the unactivated output of the convolutional layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
    A_prev -- is a numpy.ndarray containing the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
    W -- is a numpy.ndarray containing the kernels for the convolution
            kh is the filter height
            kw is the filter width
    b -- is a numpy.ndarray containing the biases applied to the convolution
    padding -- is a string that is either same or valid
    stride -- is a tuple containing the strides for the convolution
            sh is the stride for the height
            sw is the stride for the width

    Returns:
    dA_prev -- the previous layer partial derivative
    dW -- the kernels partial derivative
    db -- the biases partial derivative
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    h, w = h_new, w_new

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    if padding == "same":

        ph = int(((h - 1) * sh + kh - kh % 2 - h) / 2 + 1)
        pw = int(((w - 1) * sw + kw - kw % 2 - w) / 2 + 1)

    elif padding == "valid":
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        constant_values=0, mode='constant')
    dA_prev_pad = np.pad(
        dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        constant_values=0, mode='constant')

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vs = h * sh
                    ve = h * sh + kh
                    hs = w * sw
                    he = w * sw + kw

                    a_slice = a_prev_pad[vs:ve, hs:he, :]

                    da_prev_pad[vs:ve, hs:he, :] += W[:,
                                                      :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        if (padding == 'valid'):
            dA_prev[i, :, :, :] = da_prev_pad
        elif (padding == 'same'):
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw, :]
    return dA_prev, dW, db
