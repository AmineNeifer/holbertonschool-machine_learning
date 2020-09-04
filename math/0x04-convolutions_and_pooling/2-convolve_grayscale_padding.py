#!/usr/bin/env python3


""" contains a function to convolve a matrix with padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ Performs a same convolution on grayscale images

        @images: numpy array (m, h, w) contains m grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
        @kernel: numpy array (kh, kw) contains the kernel for the convo
                kh is the height of the kernel
                kw is the width of the kernel
        @padding is a tuple of (ph, pw)
                ph is the padding for the height of the image
                pw is the padding for the width of the image
                the image should be padded with 0’s

        Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    p_l = pw
    p_r = pw
    p_t = ph
    p_b = ph

    output_height = h - kh + (2 * ph) + 1
    output_width = w - kw + (2 * pw) + 1

    output = np.zeros((m, output_height, output_width))

    image_padded = np.pad(
        images, ((0, 0), (p_t, p_b), (p_l, p_r)), mode="constant", constant_values=0)
    # print(images)
    # print(image_padded)
    for h in range(output_height):
        for w in range(output_width):
            output[:, h, w] = np.sum(
                kernel * image_padded[:, h: h + kh, w: w + kw], axis=(1, 2))
            #print("({} - {})".format(h, w))
            # print(output)"""
    return output
