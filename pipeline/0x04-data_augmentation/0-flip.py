#!/usr/bin/env python3
""" Contains flip_image function"""
import tensorflow as tf


def flip_image(image):
    """tf.image.flip_left_right(x)"""
    return tf.image.flip_left_right(image)
