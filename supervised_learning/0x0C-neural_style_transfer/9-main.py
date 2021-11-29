#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
NST = __import__('9-neural_style').NST


if __name__ == '__main__':
    np.random.seed(2)
    tf.random.set_seed(2)
    style_image = np.random.uniform(0, 256, size=(1000, 1000, 3))
    content_image = np.random.uniform(0, 256, size=(500, 1000, 3))

    nst = NST(style_image, content_image, 1e3, 10)
    nst.generate_image(step=2, iterations=5, lr=0.05, beta1=0.99, beta2=0.999)
