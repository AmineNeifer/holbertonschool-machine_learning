#!/usr/bin/env python3
""" Contains change_brightness function"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """ randomly changes the brightness of an image"""
    return tf.image.random_brightness(
        image, max_delta
    )
