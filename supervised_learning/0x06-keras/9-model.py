#!/usr/bin/env python3


""" Contains [save/load] functions"""
import tensorflow.keras as K


def save_model(network, filename):
    """ saves a model"""
    network.save(filename)
    return None


def load_model(filename):
    """ loads a model"""
    model = K.models.load_model(filename)
    return model
