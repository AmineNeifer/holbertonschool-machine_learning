#!/usr/bin/env python3

""" contains NST class"""
import numpy as np
import tensorflow as tf


class NST:
    """ neural style transfer class"""
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ initialization of NST"""
        if not isinstance(style_image, np.ndarray) or len(
                style_image.shape) != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or len(
                content_image.shape) != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(alpha, int) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, int) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        rescales an image such that pixels are in [0..1] and
        its largest side is 512 pixels
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        max_dims = 512
        shape = image.shape[:2]
        scale = max_dims / max(shape[0], shape[1])
        new_shape = (int(scale * shape[0]), int(scale * shape[1]))
        image = np.expand_dims(image, axis=0)
        image = tf.clip_by_value(tf.image.resize(image, new_shape, 'bicubic') / 255, 0, 1)
        return image
