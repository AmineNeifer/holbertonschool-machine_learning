#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Flatten, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
#make VAE 

def autoencoder(input_dims, hidden_layers, latent_dims):
  def sampling_func(inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dims),
                              mean=0., stddev=1.)
    return z_mean + tf.exp(z_log_var / 2) * epsilon

  def make_encoder(input_dims, hidden_layers, latent_dims):
    enc_inp = Input(shape=(input_dims,))
    x = enc_inp
    for n in hidden_layers:
      x = Dense(n, activation='relu')(x)
    z_mean = Dense(latent_dims)(x)
    z_log_var = Dense(latent_dims)(x)
    z = sampling_layer([z_mean, z_log_var])
    return Model(inputs=enc_inp, outputs=[z, z_mean, z_log_var],
                name="encoder")  

  def make_decoder(input_dims, hidden_layers, latent_dims):
    decoder_input = Input(shape=(latent_dims,))
    x = decoder_input
    for n in hidden_layers[::-1]:
      x = Dense(n, activation='relu')(x)
    x = Dense(input_dims, activation='sigmoid')(x)
    return Model(decoder_input, x, name="decoder")  
  
  #initialize a sampling layer
  sampling_layer = Lambda(sampling_func, output_shape=(latent_dims,)) 
  
  encoder = make_encoder(input_dims, hidden_layers, latent_dims)
  decoder = make_decoder(input_dims, hidden_layers, latent_dims)

  x = Input(shape=(input_dims,))
  z, z_mean, z_log_var = encoder(x)
  #z = sampling_layer([z_mean, z_log_var])
  x_decoded_mean = decoder(z)
  vae = Model(x, x_decoded_mean) 

  #loss function :
  #we have two losses to consider  :
  #the total loss is a sum of the reconstruction loss and the KL divergence loss term
  #stands for reconstruction loss
  #we will use the binary crossentropy

  print("x is :", x)
  print("x decoded mean :", x_decoded_mean)
  
  r_loss = input_dims * metrics.binary_crossentropy(x, x_decoded_mean) 
  print("r_loss", r_loss)
  kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - z_mean **2 - tf.math.exp(z_log_var), axis=-1)
  print("kl_loss", kl_loss)
  total_loss = tf.reduce_mean(r_loss + kl_loss)

  vae.add_loss(total_loss)
  vae.compile(optimizer="adam")
  
  
  return encoder, decoder, vae
