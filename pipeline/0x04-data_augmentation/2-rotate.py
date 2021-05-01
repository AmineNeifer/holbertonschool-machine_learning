#!/usr/bin/env python3
""" Contains rotate_image function"""
import tensorflow as tf


def rotate_image(image):
    """ Rotates an image by 90 degrees counter-clockwise"""
    return tf.image.rot90(image, k=1)
