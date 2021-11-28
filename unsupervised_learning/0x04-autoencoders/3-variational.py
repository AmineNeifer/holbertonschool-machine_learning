#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from scipy.stats import norm
from tensorflow.keras.layers import Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K



def autoencoder(input_dims, hidden_layers, latent_dims):
  if hidden_layers is None:
    hidden_layers = []

  #encoder 
  Ei = Input(shape=(input_dims,))
  x = Ei
  for l in hidden_layers:
    x = Dense(l, activation='relu')(x)

  print("shape of x: ", K.int_shape(x)[1:])  

  mu = Dense(latent_dims)(x)
  log_var = Dense(latent_dims)(x)

  
  def sample(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon

  Eo = Lambda(sample)([mu, log_var])
  encoder = Model(inputs=Ei, outputs=[Eo, mu, log_var])  

  #Decoder 
  Di = Input(shape=(latent_dims,))
  x = Di
  for l in reversed(hidden_layers):
    x = Dense(l, activation='relu')(x)
  Do = Dense(input_dims, activation="sigmoid")(x)
  decoder = Model(inputs=Di, outputs=Do)
  
  #autoencoder
  auto = Model(inputs=Ei, outputs=decoder(encoder(Ei)[0]))

  #Loss:
  optimizer = tf.keras.optimizers.Adam()

  def vae_kl_loss(x, x_decoded):
    kl_loss = -0.5 * (1 + log_var - K.square(mu) - K.exp(log_var)) #shape (None, 2)
    kl_loss = K.sum(kl_loss, axis = 1)
    return kl_loss
  
  def vae_r_loss(x, x_decoded):
    #print(K.shape(K.binary_crossentropy(x, x_decoded)))
    r_loss = K.sum(K.binary_crossentropy(x, x_decoded), axis=1)
    #r_loss = K.mean(K.square(x - x_decoded), axis = 1)
    print(r_loss.shape)
    return r_loss 
    
  def total_loss(x, x_decoded):
    #x_train ( 6000, 784)
    r_loss = vae_r_loss(x, x_decoded) 
    kl_loss = vae_kl_loss(x, x_decoded)
    return r_loss + kl_loss

  auto.compile(optimizer="adam", loss=total_loss)
  return encoder, decoder, auto  