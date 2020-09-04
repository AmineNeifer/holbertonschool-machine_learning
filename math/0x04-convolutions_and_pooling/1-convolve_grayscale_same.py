#!/usr/bin/env python3


""" contains a function to same convolve a matrix"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Performs a same convolution on grayscale images

        @images: numpy array (m, h, w) contains m grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
        @kernel: numpy array (kh, kw) contains the kernel for the convo
                kh is the height of the kernel
                kw is the width of the kernel

        Returns: a numpy.ndarray containing the convolved images"""
    image_padded = np.pad(images, ((0, 0), (1, 1), (1, 1)),
                          mode="constant", constant_values=0)
    m, h, w = image_padded.shape
    kh, kw = kernel.shape

    output_height = int(h - kh + 1)
    output_width = int(w - kw + 1)

    output = np.zeros((m, output_height, output_width))
    for h in range(output_height):
        for w in range(output_width):
            output[:, h, w] = np.sum(
                kernel * image_padded[:, h: h + kh, w: w + kw], axis=(1, 2))
    return output
