#!/usr/bin/env python3


""" contains a function to valid convolve a matrix"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_valid(images, kernel):
    """ Performs a valid convolution on grayscale images

        images is a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
                kh is the height of the kernel
                kw is the width of the kernel

        Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_height = int(ceil(h - kh + 1))
    output_width = int(ceil(w - kw + 1))

    output = np.zeros((m, output_height, output_width))
    for h in range(output_height):
        for w in range(output_width):
            output[:, h, w] = np.sum(
                kernel * images[:, h: h + kh, w: w + kw], axis=(1, 2))
    return output
