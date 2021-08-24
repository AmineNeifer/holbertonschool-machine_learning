#!/usr/bin/env python3

""" Docs lela """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def pr_gradient(step_number, step_cost, step_accuracy):
    """ Print gradient descent within an epoch """
    print("\tStep {}:".format(step_number))
    print("\t\tCost: {}".format(step_cost))
    print("\t\tAccuracy: {}".format(step_accuracy))


def pr_ep(ephoc, train_cost, train_accuracy, valid_cost, valid_accuracy):
    """ Print before the first epoch and after
        every subsequent epoch
    """
    print("After {} epochs:".format(ephoc))
    print("\tTraining Cost: {}".format(train_cost))
    print("\tTraining Accuracy: {}".format(train_accuracy))
    print("\tValidation Cost: {}".format(valid_cost))
    print("\tValidation Accuracy: {}".format(valid_accuracy))


def calculates(sess, loss, accuracy, train_dict, valid_dict):
    """ Calculates train and valid accuracy and cost values """
    train_cost = sess.run(loss, train_dict)
    train_accuracy = sess.run(accuracy, train_dict)
    valid_cost = sess.run(loss, valid_dict)
    valid_accuracy = sess.run(accuracy, valid_dict)
    return train_cost, train_accuracy, valid_cost, valid_accuracy


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ trains a loaded neural network model
        using mini-batch gradient descent
        @X_train: is a numpy.ndarray of shape (m, 784)
                  containing the training data
            @m: is the number of data points
            @784: is the number of input features
        @Y_train: is a one-hot numpy.ndarray of shape (m, 10)
                  containing the training labels
            @10: is the number of classes the model should classify
        @X_valid: is a numpy.ndarray of shape (m, 784)
                  containing the validation data
        @Y_valid: is a one-hot numpy.ndarray of shape (m, 10)
                  containing the validation labels
        @batch_size: is the number of data points in a batch
        @epochs: is the number of times the training should
                 pass through the whole dataset
        @load_path is the path from which to load the model
        @save_path: is the path to where the model should
                    be saved after training
        Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            feed_dict_train = {x: X_train, y: Y_train}
            feed_dict_valid = {x: X_valid, y: Y_valid}

            train_cost, train_accuracy, valid_cost, valid_accuracy = \
                calculates(sess, loss, accuracy,
                           feed_dict_train, feed_dict_valid)

            pr_ep(epoch, train_cost, train_accuracy,
                  valid_cost, valid_accuracy)

            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                i = 0
                step = 1
                while i < X_shuffled.shape[0]:
                    if i + batch_size > X_shuffled.shape[0]:
                        end = X_shuffled.shape[0]
                    else:
                        end = i + batch_size
                    x_mini = X_shuffled[i: end]
                    y_mini = Y_shuffled[i: end]
                    feed_dict_train = {x: x_mini, y: y_mini}
                    sess.run(train_op, feed_dict_train)
                    if step % 100 == 0:
                        _, _, valid_cost, valid_accuracy =\
                            calculates(sess, loss, accuracy,
                                       feed_dict_train, feed_dict_train)
                        pr_gradient(step, valid_cost, valid_accuracy)
                    i += batch_size
                    step += 1
        x = tf.add_to_collection('x', x)
        y = tf.add_to_collection('y', y)
        accuracy = tf.add_to_collection('accuracy', accuracy)
        loss = tf.add_to_collection('loss', loss)
        train_op = tf.add_to_collection('train_op', train_op)
        return saver.save(sess, save_path)