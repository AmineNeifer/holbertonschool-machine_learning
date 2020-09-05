#!/usr/bin/env python3


""" contains a function to convolve a matrix"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ Performs a same convolution on grayscale images

        @images: numpy array (m, h, w) contains m grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
        @kernel: numpy array (kh, kw) contains the kernel for the convo
                kh is the height of the kernel
                kw is the width of the kernel
        @padding: a tuple of (ph, pw)
                ph is the padding for the height of the image
                pw is the padding for the width of the image
                the image should be padded with 0’s
        @stride: a tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image
        Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == "same":
        output_height = int(h // sh)
        output_width = int(w // sw)

        pad_h = max((output_height - 1) * sh + kh - h, 0)
        pad_w = max((output_width - 1) * sw + kw - w, 0)
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        output = np.zeros((m, output_height, output_width))

        image_padded = np.zeros((m, h + pad_h, w + pad_w))
        image_padded[:, pad_t: - pad_b, pad_l: -pad_r] = np.copy(images)
    elif padding == "valid":
        output_height = int(np.ceil((h - kh + 1) / sh))
        output_width = int(np.ceil((w - kw + 1) / sw))
        image_padded = np.copy(images)
        output = np.zeros((m, output_height, output_width))
    
    elif isinstance(padding, tuple):
        ph, pw = padding
        p_l = pw
        p_r = pw
        p_t = ph
        p_b = ph

        output_height = (h - kh + (2 * ph) + 1) // sh
        output_width = (w - kw + (2 * pw) + 1) // sw

        output = np.zeros((m, output_height, output_width))

        image_padded = np.pad(
            images, ((0, 0), (p_t, p_b), (p_l, p_r)),
            mode="constant", constant_values=0)

    for h in range(output_height):
        for w in range(output_width):
            h_s = h * sh
            w_s = w * sw
            output[:, h, w] = np.sum(
                kernel * image_padded[:, h_s: h_s + kh, w_s: w_s + kw],
                axis=(1, 2))
    return output
