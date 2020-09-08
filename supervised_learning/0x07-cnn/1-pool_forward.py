#!/usr/bin/env python3


""" contains pooling forward prop function"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Performs forward prop over a convolutional layer of a neural network:

        @A_prev:(m, h_prev, w_prev, c_prev) contains output of the prev layer
                m is the number of examples
                h_prev is the height of the previous layer
                w_prev is the width of the previous layer
                c_prev is the number of channels in the previous layer
        @kernel_shape: tuple (kh, kw) contains size of the kernel
                kh is the filter height
                kw is the filter width
        @stride is a tuple of (sh, sw) contains the strides for the convolution
                sh is the stride for the height
                sw is the stride for the width
        @mode is a string containing either max or avg, respectively

        Returns: the output of the convolutional layer"""
    return pool(A_prev, kernel_shape, stride, mode)


def pool(images, kernel_shape, stride, mode='max'):
    """ Performs pooling on images
        @images:a with shape (m, h, w, c) containing multiple images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
                c is the number of channels in the image
        @kernel_shape: a tuple of (kh, kw) containing the kernel shape
                kh is the height of the kernel
                kw is the width of the kernel
        @stride: a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
        @mode: indicates the type of pooling
                max indicates max pooling
                avg indicates average pooling
        Returns: a numpy.ndarray containing the pooled images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = int((h - kh) / sh + 1)
    output_w = int((w - kw) / sw + 1)

    output = np.zeros((m, output_h, output_w, c))
    for h in range(output_h):
        for w in range(output_w):
            if mode == 'max':
                output[:, h, w, :] = np.max(
                    images[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :],
                    axis=(1, 2))
            else:
                output[:, h, w, :] = np.mean(
                    images[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :],
                    axis=(1, 2))
    return output
