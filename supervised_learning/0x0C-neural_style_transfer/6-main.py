#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
NST = __import__('6-neural_style').NST


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)
    style_image = np.random.uniform(0, 256, size=(1000, 1000, 3))
    content_image = np.random.uniform(0, 256, size=(500, 1000, 3))

    nst = NST(style_image, content_image)
    content_output = nst.model(tf.keras.applications.vgg19.preprocess_input(nst.content_image * 255))[-1]
    print(nst.content_cost(content_output))
    content_image = tf.random.uniform(nst.content_image.shape)
    content_output = nst.model(tf.keras.applications.vgg19.preprocess_input(content_image * 255))[-1]
    print(nst.content_cost(content_output))
