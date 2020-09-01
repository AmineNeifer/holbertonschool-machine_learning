#!/usr/bin/env python3


""" contains a function to give the prediction of the model"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ return prediction of a model"""
    return network.predict(data, verbose=verbose)
