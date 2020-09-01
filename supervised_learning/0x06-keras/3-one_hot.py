#!/usr/bin/env python3


""" contains one_hot function encoder"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ one_hot encoder"""
    return K.utils.to_categorical(labels, classes)
