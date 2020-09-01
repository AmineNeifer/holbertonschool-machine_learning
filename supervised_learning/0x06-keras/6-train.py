#!/usr/bin/env python3


""" Contains a functin that trains a model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ train model using keras"""
    if (early_stopping and validation_data):
        callback = [K.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience)]
    else:
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
