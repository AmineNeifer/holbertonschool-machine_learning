#!/usr/bin/env python3


""" contains convo forward prop function"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Performs forward prop over a convo layer of a neural network:

        @A_prev: (m, h_prev, w_prev, c_prev) contains the output of prev layer
                m is the number of examples
                h_prev is the height of the previous layer
                w_prev is the width of the previous layer
                c_prev is the number of channels in the previous layer
        @W: (kh, kw, c_prev, c_new) containing the kernels for the convolution
                kh is the filter height
                kw is the filter width
                c_prev is the number of channels in the previous layer
                c_new is the number of channels in the output
        @b: (1, 1, 1, c_new) containing the biases
        @activation is an activation function applied to the convolution
        @padding:string that is either same or valid
        @stride is a tuple of (sh, sw) containing the strides for the convo
                sh is the stride for the height
                sw is the stride for the width"

        Returns: the output of the convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    Z = convolve(A_prev, W, padding, stride) + b
    A = activation(Z)
    return A


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ Performs a convolution on images using multiple kernels
        @images: numpy array (m, h, w, c) contains m grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
                c is the number of channels in the image
        @kernel: numpy array (kh, kw, c, nC) contains the kernel for the convo
                kh is the height of the kernel
                kw is the width of the kernel
                nc is the number of kernels
        @padding: a tuple of (ph, pw), 'same' or 'valid'
                ph is the padding for the height of the image
                pw is the padding for the width of the image
                the image should be padded with 0s
        @stride: a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
        Returns: a numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride
    if padding == "same":
        ph, pw = 0, 0

        ph = int(((h - 1) * sh + kh - h) / 2 + 1)
        pw = int(((w - 1) * sw + kw - w) / 2 + 1)

        image_padded = np.zeros((m, h + 2 * ph, w + 2 * pw, c))
        image_padded[:, ph:h + ph, pw:w + pw, :] = images

    elif padding == "valid":
        ph, pw = 0, 0
        output_height = int(np.ceil((h - kh + 1) / sh))
        output_width = int(np.ceil((w - kw + 1) / sw))
        image_padded = np.copy(images)

    output_height = int((h - kh + (2 * ph)) // sh + 1)
    output_width = int((w - kw + (2 * pw)) // sw + 1)

    output = np.zeros((m, output_height, output_width, nc))
    for n in range(nc):
        for h in range(output_height):
            for w in range(output_width):
                h_s = h * sh
                w_s = w * sw
                output[:, h, w, n] = np.sum(
                    image_padded[:, h_s: h_s + kh, w_s: w_s +
                                 kw, :] * kernels[:, :, :, n],
                    axis=(1, 2, 3))
        output.sum(axis=1)
    return output
