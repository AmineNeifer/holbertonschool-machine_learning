#!/usr/bin/env python3


""" Contains a functin that trains a model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """ trains a model using keras"""
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, shuffle=shuffle, verbose=verbose)
    return history
