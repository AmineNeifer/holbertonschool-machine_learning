#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

create_layer = __import__('1-create_layer').create_layer

np.random.seed(1)
a, nx, nl = np.random.randint(1, 40, 3)
X = np.random.randn(a, nx)

tf.set_random_seed(0)
x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
l = create_layer(x, nl, tf.nn.sigmoid)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(np.array2string(sess.run(l, feed_dict={x:X}), precision=5))