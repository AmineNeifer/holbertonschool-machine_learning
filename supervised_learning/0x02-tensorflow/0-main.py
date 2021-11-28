#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders

np.random.seed(0)
a, b, nx, c = np.random.randint(1, 30, 4)
X = np.random.randn(a, nx)
Y = np.random.randn(b, c)

x, y = create_placeholders(nx, c)

with tf.Session() as sess:
    print(sess.run((x, y), feed_dict={x:X, y:Y}))