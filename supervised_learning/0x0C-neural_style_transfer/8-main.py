#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
NST = __import__('8-neural_style').NST


if __name__ == '__main__':
    np.random.seed(2)
    style_image = np.random.uniform(0, 256, size=(1000, 1000, 3))
    content_image = np.random.uniform(0, 256, size=(500, 1000, 3))

    nst = NST(style_image, content_image)

    try:
        nst.compute_grads(nst.content_image.numpy())
    except TypeError as e:
        print(str(e))
    try:
        nst.compute_grads(nst.content_image[0])
    except TypeError as e:
        print(str(e))
    try:
        nst.compute_grads(tf.random.uniform((1, 5, 10, 3)))
    except TypeError as e:
        print(str(e))
    nst.compute_grads(nst.content_image)
    nst.compute_grads(tf.Variable(nst.content_image))
