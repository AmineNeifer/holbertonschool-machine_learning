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
                style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or len(
                content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if (type(alpha) is not int and type(alpha) is not float) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if (type(beta) is not int and type(beta) is not float) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

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
        image = tf.clip_by_value(tf.image.resize(
            image, new_shape, 'bicubic') / 255, 0, 1)
        return image

    def load_model(self):
        """Creates the model used to calculate the cost"""
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        x = vgg.input
        model_outputs = []
        content_output = None
        for layer in vgg.layers[1:]:
            if "pool" in layer.name:
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    name=layer.name)(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    model_outputs.append(x)
                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False
        model_outputs.append(content_output)
        model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrices"""
        if not (isinstance(input_layer, tf.Tensor)
                or isinstance(input_layer, tf.Variable)) \
                or input_layer.shape.ndims != 4:
            raise TypeError('input_layer must be a tensor of rank 4')
        # input_layer: containing the layer output
        # whose gram matrix should be calculated
        _, nh, nw, _ = input_layer.shape.dims
        G = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        return G / tf.cast(nh * nw, tf.float32)

    def generate_features(self):
        """Extracts the features used to calculate neural style cost"""
        preprocessed_s = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_c = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_features = self.model(preprocessed_s)[:-1]
        self.content_feature = self.model(preprocessed_c)[-1]
        self.gram_style_features = [self.gram_matrix(
            style_feature) for style_feature in style_features]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer"""
        if not (isinstance(style_output, tf.Tensor)
                or isinstance(style_output, tf.Variable)) \
                or style_output.shape.ndims != 4:
            raise TypeError('style_output must be a tensor of rank 4')
        m, _, _, nc = style_output.shape.dims
        if not (isinstance(gram_target, tf.Tensor)
                or isinstance(gram_target, tf.Variable)) \
                or gram_target.shape.dims != [m, nc, nc]:
            raise TypeError(
                'gram_target must be a tensor of shape [{}, {}, {}]'.
                format(m, nc, nc))

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_sum(tf.square(gram_style - gram_target)) \
            / tf.square(tf.cast(nc, tf.float32))

    def style_cost(self, style_outputs):
        """Calculates the style cost for generated image"""

        if type(style_outputs) is not list or \
                len(style_outputs) != len(self.style_layers):
            raise TypeError('style_outputs must be a list with a length of {}'.
                            format(
                                len(self.style_layers)))
        J_style = tf.add_n([
            self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]) for i in range(len(style_outputs))
        ])
        J_style /= tf.cast(len(style_outputs), tf.float32)
        return J_style

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image"""
        if not (isinstance(content_output, tf.Tensor) or
                isinstance(content_output, tf.Variable)) or \
                content_output.shape.dims != self.content_feature.shape.dims:
            raise TypeError('content_output must be a tensor of shape {}'.
                            format(
                                self.content_feature.shape))
        _, nh, nw, nc = content_output.shape.dims
        return tf.reduce_sum(tf.square(content_output - self.content_feature))\
            / tf.cast(nh * nw * nc, tf.float32)

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image"""
        if not (isinstance(generated_image, tf.Tensor) or
                isinstance(generated_image, tf.Variable)) or \
                generated_image.shape.dims != self.content_image.shape.dims:
            raise TypeError('generated_image must be a tensor of shape {}'.
                            format(
                                self.content_image.shape))
        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        model_outputs = self.model(preprocessed)
        style_outputs = [style_layer for style_layer in model_outputs[:-1]]
        content_output = model_outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J = (self.alpha * J_content) + (self.beta * J_style)
        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image"""
        if not (isinstance(generated_image, tf.Tensor) or
                isinstance(generated_image, tf.Variable)) or \
                generated_image.shape.dims != self.content_image.shape.dims:
            raise TypeError('generated_image must be a tensor of shape {}'.
                            format(
                                self.content_image.shape))
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J, J_content, J_style = self.total_cost(generated_image)
        grads = tape.gradient(J, generated_image)
        return grads, J, J_content, J_style
