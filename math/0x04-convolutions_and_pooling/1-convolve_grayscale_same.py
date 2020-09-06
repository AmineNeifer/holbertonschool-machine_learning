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
    # image_padded = np.pad(images, ((0, 0), (1, 1), (1, 1)),
    #                      mode="constant", constant_values=0)
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_height = h
    output_width = w

    if kh % 2 == 1:
        pad_h = (kh - 1) // 2
    else:
        pad_h = kh // 2
    if kw % 2 == 1:
        pad_w = (kw - 1) // 2
    else:
        pad_w = kw // 2

    output = np.zeros((m, output_height, output_width))

    image_padded = np.zeros((m, h + 2 * pad_h, w + 2 * pad_w))
    image_padded[:, pad_h:h + pad_h, pad_w:w + pad_w] = images.copy()
    print(images[0])
    print(image_padded[0])
    print('-------------------')
    print(kernel)
    for h in range(output_height):
        for w in range(output_width):
            output[:, h, w] = np.sum(
                kernel * image_padded[:, h: h + kh, w: w + kw], axis=(1, 2))
    print(output[0])

    return output
