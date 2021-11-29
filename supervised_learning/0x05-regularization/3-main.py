#!/usr/bin/env python3

import numpy as np
import os
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

# Reproducibility
def set_seed(seed=31415):
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.set_random_seed(seed)

    np.random.seed(seed)
set_seed(6)

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    oh = np.zeros((classes, m))
    oh[Y, np.arange(m)] = 1
    return oh

m = np.random.randint(1000, 2000)
c = 10
lib= np.load('./MNIST.npz')
X = lib['X_train'][:m].reshape((m, -1))
Y = one_hot(lib['Y_train'][:m], c).T
n0 = X.shape[1]
n1, n2 = np.random.randint(10, 1000, 2)
lam = np.random.uniform(0.01)
x = tf.placeholder(tf.float32, (None, n0))
y = tf.placeholder(tf.float32, (None, c))
a1 = l2_reg_create_layer(x, n1, tf.nn.tanh, lam)
a2 = l2_reg_create_layer(a1, n2, tf.nn.sigmoid, lam)
y_pred = l2_reg_create_layer(a2, c, None, 0.)
cost = tf.losses.softmax_cross_entropy(y, y_pred)
l2_cost = cost + tf.losses.get_regularization_losses()
train_op = tf.train.AdamOptimizer().minimize(l2_cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        sess.run(train_op, feed_dict={x: X, y: Y})
        print(sess.run(l2_cost, feed_dict={x: X, y: Y}))
