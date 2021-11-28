#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

NST = __import__('1-neural_style').NST


if __name__ == '__main__':
    np.random.seed(3)
    style_image = np.random.uniform(0, 256, size=(1000, 1000, 3))
    content_image = np.random.uniform(0, 256, size=(500, 1000, 3))

    nst = NST(style_image, content_image)
    print(nst.model.get_weights())
