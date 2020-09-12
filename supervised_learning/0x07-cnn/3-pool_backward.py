#!/usr/bin/env python3

"""contains pooling backward function"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network:

    Arguments:
    dA -- partial derivatives with respect to the output of the pooling layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
    A_prev -- the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
    kernel_shape -- contains the size of the kernel for the pooling
            kh is the kernel height
            kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
            sh is the stride for the height
            sw is the stride for the width
    mode -- is a string containing either max or avg

    Returns:
    the partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, n_c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(n_c):
                    vs = h * sh
                    ve = h * sh + kh
                    hs = w * sw
                    he = w * sw + kw

                    if mode == "max":
                        a_prev_slice = a_prev[vs:ve, hs:he, c]
                        mask = create_mask(a_prev_slice)
                        dA_prev[i, vs:ve, hs:he,
                                c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        dA_prev[i, vs:ve, hs:he,
                                c] += distribute_value(da, kernel_shape)
    return dA_prev


def create_mask(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window
    contains a True at the position corresponding to the max entry of x.
    """
    mask = (x == np.max(x))
    return mask


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix
             for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    (n_h, n_w) = shape
    average = dz / (n_h * n_w)
    return np.ones(shape) * average
