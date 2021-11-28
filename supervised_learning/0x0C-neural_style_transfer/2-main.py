#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

NST = __import__('2-neural_style').NST

if __name__ == '__main__':
    np.random.seed(0)
    input_layer = np.random.randn(320, 280, 3)
    try:
        gram_matrix = NST.gram_matrix(input_layer[np.newaxis, :])
    except TypeError as e:
        print(str(e))
    try:
        gram_matrix = NST.gram_matrix(tf.constant(input_layer, dtype=tf.float32))
    except TypeError as e:
        print(str(e))
