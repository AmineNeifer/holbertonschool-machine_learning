#!/usr/bin/env python3


""" Contains [save/load] functions"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model weights"""
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(network, filename):
    """ loads a model weights"""
    with open(filename, "r") as f:
        load = K.models.model_from_json(f.read())
    return None
