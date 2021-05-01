#!/usr/bin/env python3
""" Contains shear_image function"""

import tensorflow as tf


def shear_image(image, intensity):
    """ Randomly shears an image"""
    # return image
    return tf.keras.preprocessing.image.apply_affine_transform(
        image.numpy(),
        shear=intensity/10
    )
