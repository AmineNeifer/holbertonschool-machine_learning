#!/usr/bin/env python3

""" contains train_mini_batch function"""

import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
tf.disable_eager_execution()


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    X_train: np.ndarray of shape (m, 784) containing the training data
        m: the number of data points
        784: the number of input features
    Y_train: one-hot np.ndarray of shape (m, 10) containing the training labels
        10: the number of classes the model should classify
    X_valid: np.ndarray of shape (m, 784) containing the validation data
    Y_valid: one-hot np.ndarray of shape (m, 10) contains the validation labels
    batch_size: the number of data points in a batch
    epochs: the nb of times the training should pass through the whole dataset
    load_path: the path from which to load the model
    save_path: the path to where the model should be saved after training
    Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs):
            print('After {} epochs:'.format(i))
            train_cost, train_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_train, y: Y_train})
            print('\tTraining Cost: {}'.format(train_cost))
            print('\tTraining Accuracy: {}'.format(train_accuracy))

            valid_cost, valid_accuracy = sess.run(
                (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
            print('\tValidation Cost: {}'.format(valid_cost))
            print('\tValidation Accuracy: {}'.format(valid_accuracy))

            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

            for j in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffle[j: j + batch_size]
                Y_batch = Y_shuffle[j: j + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step = (j // batch_size + 1)
                if not(step % 100):
                    cost, acc = sess.run((loss, accuracy), feed_dict={
                                         x: X_batch, y: Y_batch})
                    print('\tStep: {}'.format(j // batch_size + 1))
                    print('\t\tCost: {}'.format(cost))
                    print('\t\tAccuracy: {}'.format(acc))
        print('After {} epochs:'.format(epochs))
        train_cost, train_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_train, y: Y_train})
        print('\tTraining Cost: {}'.format(train_cost))
        print('\tTraining Accuracy: {}'.format(train_accuracy))

        valid_cost, valid_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
        print('\tValidation Cost: {}'.format(valid_cost))
        print('\tValidation Accuracy: {}'.format(valid_accuracy))
        return saver.save(sess, save_path)
