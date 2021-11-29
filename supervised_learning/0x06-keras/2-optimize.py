#!/usr/bin/env python3


""" contains optimaize_mode funct"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ optimizes the model"""
    Adam = K.optimizers.Adam
    network.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=alpha,
                                   beta_1=beta1, beta_2=beta2),
                    metrics=['acc'])
    return None
