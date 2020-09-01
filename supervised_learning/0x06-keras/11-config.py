#!/usr/bin/env python3


""" Contains [save/load] functions"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a modelc configs"""
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(network, filename):
    """ loads a model configs"""
    with open(filename, "r") as f:
        r = f.read()
        load = K.models.model_from_json(r)
    return None
