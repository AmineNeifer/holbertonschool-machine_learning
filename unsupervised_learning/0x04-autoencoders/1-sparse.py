#!/usr/bin/env python3

""" contains sparse autoencoder implementation"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates sparse autoencoder

    @input_dims is an integer containing the dimensions of the model input
    @hidden_layers: list containing the n° of nodes for eac
                    hidden layer in the encoder, respectivel
        -the hidden layers should be reversed for the decoder
    @latent_dims: int containing the dims of the latent space representation
    @lambtha: regularization parameter used for L1 regularization
    Returns: encoder, decoder, auto
        @encoder is the encoder model
        @decoder is the decoder model
        @auto is the full autoencoder model
    """
    Input = keras.Input
    Dense = keras.layers.Dense
    Model = keras.Model
    regularizers = keras.regularizers

    input_img = Input((input_dims,))
    encoded = input_img
    for hidden_layer in hidden_layers:
        encoded = Dense(hidden_layer, activation='relu')(encoded)
    encoded = Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=regularizers.l1(lambtha))(encoded)

    encoded_img = Input((latent_dims,))
    decoded = encoded_img
    for hidden_layer in reversed(hidden_layers):
        decoded = Dense(hidden_layer, activation='relu')(decoded)
    decoded = Dense(input_dims, activation="sigmoid")(decoded)

    encoder = Model(input_img, encoded)
    decoder = Model(encoded_img, decoded)
    out_decoder = decoder(encoder(input_img))
    auto = Model(input_img, out_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto
