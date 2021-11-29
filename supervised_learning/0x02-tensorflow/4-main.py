#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

calculate_loss = __import__('4-calculate_loss').calculate_loss

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

np.random.seed(1)
m, c = np.random.randint(10, 40, 2)
Y = one_hot(np.random.randint(0, c, m), c)
Y_p = one_hot(np.random.randint(0, c, m), c)

tf.set_random_seed(0)
y = tf.placeholder(tf.float32, shape=(None, c))
y_pred = tf.placeholder(tf.float32, shape=(None, c))
accuracy = calculate_loss(y, y_pred)
with tf.Session() as sess:
    print(sess.run(accuracy, feed_dict={y:Y, y_pred:Y_p}))
