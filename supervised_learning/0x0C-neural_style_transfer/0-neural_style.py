#!/usr/bin/env python3

""" contains NST class"""
import numpy as np
import cv2
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
        if alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if beta < 0:
            raise TypeError('beta must be a non-negative number')
        tf.enable_eager_execution()
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
        h, w, _ = image.shape
        if h > w:
            new_h = 512
            scale = 512 / h
            new_w = w * scale
        else:
            new_w = 512
            scale = 512 / w
            new_h = h * scale
        new_h = int(new_h)
        new_w = int(new_w)

        dim = (int(new_h), int(new_w))
        image = tf.reshape(image, [1, h, w, 3])
        image = tf.image.resize_bicubic(image, dim)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image
