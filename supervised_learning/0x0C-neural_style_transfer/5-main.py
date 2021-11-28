#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
NST = __import__('5-neural_style').NST


if __name__ == '__main__':
    np.random.seed(2)
    style_image = np.random.uniform(0, 256, size=(1000, 1000, 3))
    content_image = np.random.uniform(0, 256, size=(500, 1000, 3))

    nst = NST(style_image, content_image)
    style_outputs = nst.model(tf.keras.applications.vgg19.preprocess_input(nst.content_image * 255))[:-1]
    try:
        nst.style_cost(tuple(style_outputs))
    except TypeError as e:
        print(str(e))
    try:
        nst.style_cost(style_outputs[:-1])
    except TypeError as e:
        print(str(e))
