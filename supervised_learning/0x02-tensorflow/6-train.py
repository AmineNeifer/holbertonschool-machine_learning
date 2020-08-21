#!/usr/bin/env python3


""" contains train funct"""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        layer_sizes,
        activations,
        alpha,
        iterations,
        save_path="/tmp/model.ckpt"):
    """ train funct"""
    x, y = create_placeholders(X_valid.shape[1], Y_valid.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            if (i % 100 == 0 or i == iterations):
                print("After {} iterations".format(i))
                print(
                    "\tTraining Cost: {}".format(
                        sess.run(
                            loss,
                            feed_dict={
                                x: X_train,
                                y: Y_train})))
                print(
                    "\tTraining Accuracy: {}".format(
                        sess.run(
                            accuracy,
                            feed_dict={
                                x: X_train,
                                y: Y_train})))
                print(
                    "\tTraining Cost: {}".format(
                        sess.run(
                            loss,
                            feed_dict={
                                x: X_valid,
                                y: Y_valid})))
                print(
                    "\tTraining Accuracy: {}".format(
                        sess.run(
                            accuracy,
                            feed_dict={
                                x: X_valid,
                                y: Y_valid})))

            if (i == iterations):
                return saver.save(sess, save_path)
            sess.run(train_op, {x: X_train, y: Y_train})
