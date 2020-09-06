#!/usr/bin/env python3


""" contains a function to convolve a matrix"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ Performs a same convolution on grayscale images

        @images: numpy array (m, h, w, c) contains m grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
                c is the number of channels in the image
        @kernel: numpy array (kh, kw, c) contains the kernel for the convo
                kh is the height of the kernel
                kw is the width of the kernel
        @padding: a tuple of (ph, pw), 'same' or 'valid'
                ph is the padding for the height of the image
                pw is the padding for the width of the image
                the image should be padded with 0’s
        @stride: a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
        Returns: a numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
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

    elif isinstance(padding, tuple):
        ph, pw = padding
        p_l = pw
        p_r = pw
        p_t = ph
        p_b = ph

        image_padded = np.pad(
            images, ((0, 0), (p_t, p_b), (p_l, p_r), (0, 0)),
            mode="constant", constant_values=0)

    output_height = int((h - kh + (2 * ph)) // sh + 1)
    output_width = int((w - kw + (2 * pw)) // sw + 1)

    output = np.zeros((m, output_height, output_width))
    for h in range(output_height):
        for w in range(output_width):
            h_s = h * sh
            w_s = w * sw
            output[:, h, w] = np.sum(
                image_padded[:, h_s: h_s + kh, w_s: w_s + kw, :] * kernel,
                axis=(1, 2, 3))
    return output
