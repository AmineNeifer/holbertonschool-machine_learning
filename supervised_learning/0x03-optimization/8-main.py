#!/usr/bin/env python3

import numpy as np
import os
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

create_RMSProp_op = __import__('8-RMSProp').create_RMSProp_op


# Reproducibility
def set_seed(seed=31415):
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    np.random.seed(seed)
set_seed(9)

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

# set variables
m1 = np.random.randint(10, 50)
m2 = np.random.randint(m1, 100)
c = 10
a, b = np.random.uniform(0.01, size=2)
lib= np.load('./MNIST.npz')
X_test = lib['X_test'][m1:m2].reshape((m2 - m1, -1))
Y_test = one_hot(lib['Y_test'][m1:m2], c)
nx = X_test.shape[1]

tf.set_random_seed(0)
x = tf.placeholder(tf.float32, shape=(None, nx))
y = tf.placeholder(tf.float32, shape=(None, c))
layer = tf.layers.Dense(c, activation=None, kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'))
y_pred = layer(x)
loss = tf.losses.softmax_cross_entropy(y, y_pred)
train_op = create_RMSProp_op(loss, a, b, 1e-8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_op, feed_dict={y:Y_test, x:X_test})
    print(sess.run(y_pred, feed_dict={x:X_test}))
