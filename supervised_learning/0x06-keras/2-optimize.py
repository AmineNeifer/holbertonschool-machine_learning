#!/usr/bin/env python3


""" contains optimaize_mode funct"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ optimizes the model"""
    optimizer = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)

    network.compile(
        optimizer=optimizer,
        loss=K.losses.categorical_crossentropy)
