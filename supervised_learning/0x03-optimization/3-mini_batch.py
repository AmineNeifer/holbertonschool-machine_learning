#!/usr/bin/env python3

""" Docs lela """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


# def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="model.ckpt", save_path="model.ckpt"):
#    """ haja"""
    # with tf.Session() as sess:
    #     new_saver = tf.train.import_meta_graph(load_path + ".meta")
    #     new_saver.restore(sess, load_path)
    #     x = tf.get_collection('x')[0]
    #     y = tf.get_collection('y')[0]
    #     train_op = tf.get_collection('train_op')[0]
    #     loss = tf.get_collection('loss')[0]
    #     accuracy = tf.get_collection('accuracy')[0]
    #     # steps = int(np.ceil(X_train.shape[0] / 32))
    #     for epoch in range(epochs):
    #         X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
    #         print("After {} epochs:".format(epoch))
    #         print("\tTraining Cost: {}".format(
    #             sess.run(loss, feed_dict={x: X_train, y: Y_train})))
    #         print("\tTraining Accuracy: {}".format(
    #             sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
    #         print("\tTraining Cost: {}".format(
    #             sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
    #         print("\tTraining Accuracy: {}".format(
    #             sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
    #         for j in range(0, len(X_train), batch_size):
    #             X_batch = X_shuffle[j:j + batch_size]
    #             Y_batch = Y_shuffle[j:j + batch_size]
    #             sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
    #             step = j // batch_size + 1
    #             if not (step % 100):
    #                 cost, acc = sess.run((loss, accuracy), feed_dict={
    #                                      x: X_batch, y: Y_batch})
    #                 print('\tStep {}:'.format(j // batch_size + 1))
    #                 print('\t\tCost: {}'.format(cost))
    #                 print('\t\tAccuracy: {}'.format(acc))

    #     print("After {} epochs:".format(epoch+1))
    #     print("\tTraining Cost: {}".format(
    #         sess.run(loss, feed_dict={x: X_train, y: Y_train})))
    #     print("\tTraining Accuracy: {}".format(
    #         sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
    #     print("\tTraining Cost: {}".format(
    #         sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
    #     print("\tTraining Accuracy: {}".format(
    #         sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
    #     return new_saver.save(sess, save_path)

def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    '''trains a loaded neural network model using mini-batch gradient descent
    Args:
        X_train is a numpy.ndarray of shape (m, 784) containing the training
                data
            m is the number of data points
            784 is the number of input features
        Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the
                training labels
            10 is the number of classes the model should classify
        X_valid is a numpy.ndarray of shape (m, 784) containing the validation
                data
        Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the
                validation labels
        batch_size is the number of data points in a batch
        epochs is the number of times the training should pass through the
               whole dataset
        load_path is the path from which to load the model
        save_path is the path to where the model should be saved after training
    Returns: the path where the model was saved
    Important:
    The model loaded from load_path will have the following tensors / ops saved
        in it’s collection:
        x is a placeholder for the input data
        y is a placeholder for the labels
        accuracy is an op to calculate the accuracy of the model
        loss is an op to calculate the cost of the model
        train_op is an op to perform one pass of gradient descent on the model
    '''
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        train_op = tf.get_collection("train_op")[0]

        for epoch in range(epochs + 1):
            cost_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accu_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accu_val = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(accu_t))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accu_val))

            if epoch < epochs:

                X_train_s, Y_train_s = shuffle_data(X_train, Y_train)

                # MINI BATCH training:
                m = X_train.shape[0]
                for start in range(0, m, batch_size):
                    if (start + batch_size) < m:
                        X_train_m = X_train_s[start: start + batch_size]
                        Y_train_m = Y_train_s[start: start + batch_size]
                    else:
                        X_train_m = X_train_s[start: m]
                        Y_train_m = Y_train_s[start: m]

                    sess.run(train_op, {x: X_train_m, y: Y_train_m})

                    if start != 0 and (start / batch_size + 1) % 100 == 0:
                        cost_mini = sess.run(loss,
                                             feed_dict={x: X_train_m,
                                                        y: Y_train_m})
                        accu_mini = sess.run(accuracy,
                                             feed_dict={x: X_train_m,
                                                        y: Y_train_m})
                        print("\tStep {}:".format(int(start / batch_size + 1)))
                        print("\t\tCost: {}".format(cost_mini))
                        print("\t\tAccuracy: {}".format(accu_mini))

        return saver.save(sess, save_path)