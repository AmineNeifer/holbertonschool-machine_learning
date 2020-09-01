#!/usr/bin/env python3


""" Contains a functin that trains a model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """ train model using keras"""
    if validation_data is not None and len(validation_data) != 2:
        validation_data = None
    history = network.fit(data, labels, batch_size=batch_size,epochs=epochs, validation_data=validation_data,shuffle=shuffle, verbose=verbose)
    return history
