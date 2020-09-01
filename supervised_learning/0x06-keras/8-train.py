#!/usr/bin/env python3


""" Contains a functin that trains a model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """ train model using keras, with learning rate decay"""
    callback = []
    if (validation_data):
        if (early_stopping):
            callback.append(K.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience))
        if (learning_rate_decay):
            def scheduler(epoch):
                """ step decay func"""
                return alpha / (1 + decay_rate * epoch)
            c_back = K.callbacks.LearningRateScheduler(scheduler, 1)
            callback.append(c_back)
    if (save_best):
        callback.append(K.callbacks.ModelCheckpoint(filepath=filepath))
    if validation_data is False and save_best is False:
        callback = None

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=shuffle,
        verbose=verbose,
        callbacks=callback)
    return history
