#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
NST = __import__('4-neural_style').NST


if __name__ == '__main__':
    np.random.seed(2)
    style_image = np.random.uniform(0, 256, size=(1000, 1000, 3))
    content_image = np.random.uniform(0, 256, size=(500, 1000, 3))

    nst = NST(style_image, content_image)
    style_output = tf.Variable(np.random.randn(1, 53, 75, 15))
    gram_target = np.random.randn(1, 15, 15)
    try:
        nst.layer_style_cost(style_output, gram_target)
    except TypeError as e:
        print(str(e))
    gram_target = tf.constant(gram_target[0])
    try:
        nst.layer_style_cost(style_output, gram_target)
    except TypeError as e:
        print(str(e))
    gram_target = tf.constant(np.random.randn(5, 15, 15))
    try:
        nst.layer_style_cost(style_output, gram_target)
    except TypeError as e:
        print(str(e))
    gram_target = tf.constant(np.random.randn(1, 53, 53))
    try:
        nst.layer_style_cost(style_output, gram_target)
    except TypeError as e:
        print(str(e))
    gram_target = tf.constant(np.random.randn(1, 75, 75))
    try:
        nst.layer_style_cost(style_output, gram_target)
    except TypeError as e:
        print(str(e))
    gram_target = tf.constant(np.random.randn(1, 53, 75))
    try:
        nst.layer_style_cost(style_output, gram_target)
    except TypeError as e:
        print(str(e))
