#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

create_train_op = __import__('5-create_train_op').create_train_op

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

# set variables
np.random.seed(5)
m = np.random.randint(20, 101)
c = 10
a = 0.01
lib= np.load('MNIST.npz')
X_test = lib['X_test'][:m].reshape((m, -1))
Y_test = one_hot(lib['Y_test'][:m], c)
nx = X_test.shape[1]

tf.set_random_seed(0)
x = tf.placeholder(tf.float32, shape=(None, nx))
y = tf.placeholder(tf.float32, shape=(None, c))
layer = tf.layers.Dense(c, activation=None, kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'))
y_pred = layer(x)
loss = tf.losses.softmax_cross_entropy(y, y_pred)
train_op = create_train_op(loss, a)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_op, feed_dict={y:Y_test, x:X_test})
    print(np.array2string(sess.run(y_pred, feed_dict={x:X_test}), precision=5))
