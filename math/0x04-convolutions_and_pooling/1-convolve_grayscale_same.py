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

    if kh % 2 == 0:
        pad_h = int((kh - 1) / 2)
    else:
        pad_h = int(np.ceil(kh / 2))
    if kw % 2 == 0:
        pad_w = int((kw - 1) / 2)
    else:
        pad_w = int(np.ceil(kw / 2))

    p_t = pad_h // 2
    p_b = pad_h - p_t
    p_l = pad_w // 2
    p_r = pad_w - p_l

    output = np.zeros((m, output_height, output_width))

    image_padded = np.pad(images, ((0, 0), (p_t, p_b), (p_l, p_r)), mode="constant", constant_values=0)
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
