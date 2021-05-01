#!/usr/bin/env python3
""" Contains change_hue function"""
import tensorflow as tf


def change_hue(image, delta):
    """changes the hue of an image"""
    return tf.image.adjust_hue(
        image, delta
    )
