#!/usr/bin/env python3

""" Docs lela """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="model.ckpt", save_path="model.ckpt"):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + ".meta")
        new_saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        train_op = tf.get_collection('train_op')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        # steps = int(np.ceil(X_train.shape[0] / 32))
        for epoch in range(epochs):
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(
                sess.run(loss, feed_dict={x: X_train, y: Y_train})))
            print("\tTraining Accuracy: {}".format(
                sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
            print("\tTraining Cost: {}".format(
                sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
            print("\tTraining Accuracy: {}".format(
                sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
            for j in range(0, len(X_train), batch_size):
                X_batch = X_shuffle[j:j + batch_size]
                Y_batch = Y_shuffle[j:j + batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step = X_train.shape[0] // batch_size + 1
                if not (step % 100):
                    cost, acc = sess.run((loss, accuracy), feed_dict={
                                         x: X_batch, y: Y_batch})
                    print('\tStep {}:'.format(j // batch_size + 1))
                    print('\t\tCost: {}'.format(cost))
                    print('\t\tAccuracy: {}'.format(acc))

        print("After {} epochs:".format(epoch+1))
        print("\tTraining Cost: {}".format(
            sess.run(loss, feed_dict={x: X_train, y: Y_train})))
        print("\tTraining Accuracy: {}".format(
            sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
        print("\tTraining Cost: {}".format(
            sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
        print("\tTraining Accuracy: {}".format(
            sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
        return new_saver.save(sess, save_path)
