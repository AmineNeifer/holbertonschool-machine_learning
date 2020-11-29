#!/usr/bin/env python3

""" contains Variational autoencoder implementation"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates variational autoencoder

    @input_dims is an integer containing the dimensions of the model input
    @hidden_layers: list containing the n° of nodes for eac
                    hidden layer in the encoder, respectivel
        -the hidden layers should be reversed for the decoder
    @latent_dims: int containing the dims of the latent space representation

    Returns: encoder, decoder, auto
        @encoder is the encoder model
        @decoder is the decoder model
        @auto is the full autoencoder model
    """
    Input = keras.Input
    Dense = keras.layers.Dense
    Lambda = keras.layers.Lambda
    Model = keras.Model
    K = keras.backend

    inputs = Input((input_dims,))
    h = inputs
    for hidden_layer in hidden_layers:
        h = Dense(hidden_layer, activation="relu")(h)
    z_mean = Dense(latent_dims, activation=None)(h)
    z_log_sigma = Dense(latent_dims, activation=None)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims))
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling)([z_mean, z_log_sigma])

    encoder = Model(inputs, [z_mean, z_log_sigma, z])

    latent_inputs = Input((latent_dims,))

    x = latent_inputs
    for hidden_layer in reversed(hidden_layers):
        x = Dense(hidden_layer, activation='relu')(x)
    outputs = Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs)

    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs)

    # vae custom binary crossentropy loss
    def kl_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    vae.compile(optimizer="adam", loss=kl_reconstruction_loss)
    return encoder, decoder, vae
