#!/usr/bin/env python3

""" contains Vanilla autoencoder implementation"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Createes vanilla autoencoder

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
    Model = keras.Model
    input_img = Input((input_dims,))
    hidden_1 = Dense(hidden_layers[0], activation='relu')(input_img)
    hidden_2 = Dense(hidden_layers[1], activation='relu')(hidden_1)
    code = Dense(latent_dims, activation='relu')(hidden_2)
    hidden_3 = Dense(hidden_layers[1], activation='relu')(code)
    hidden_4 = Dense(hidden_layers[0], activation='relu')(hidden_3)
    output_img = Dense(input_dims, activation="sigmoid")(hidden_4)
    auto = Model(input_img, output_img)
    
    encoder = Model(input_img, hidden_2)
    decoder = Model(Input((latent_dims,)), hidden_4)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto
