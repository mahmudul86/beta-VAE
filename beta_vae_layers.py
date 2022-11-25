from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np

class Linear(layers.Layer):
  def __init__(self, name, n_width, count_iter=1, **kwargs):
    super(Linear, self).__init__(name=name, **kwargs)
    self.n_width = n_width
    self.count_iter = count_iter

  def build(self, input_shape):
    n_length = input_shape[-1]
    if self.count_iter == 1:
      self.w = self.add_weight(name='w', shape=(n_length, self.n_width),
                               initializer='random_normal',
                               dtype=tf.float32, trainable=True)
    else:
      self.w = self.add_weight(name='w', shape=(n_length, self.n_width),
                               initializer=tf.zeros_initializer(),
                               dtype=tf.float32, trainable=True)

    self.b = self.add_weight(name='b', shape=(self.n_width,),
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32, trainable=True)

  def call(self, inputs):
        return self.ls(inputs)
  
  def ls(self, inputs):
      return tf.matmul(inputs, self.w) + self.b


# two-hidden-layer neural network
class NN2(layers.Layer):
  def __init__(self, name, n_width, count_iter=1, n_out=None, **kwargs):
    super(NN2, self).__init__(name=name, **kwargs)
    self.n_width = n_width
    self.n_out = n_out
    self.count_iter = count_iter

  def build(self, input_shape):
    self.l_1 = Linear('h1', self.n_width, count_iter=self.count_iter)
    self.l_2 = Linear('h2', self.n_width, count_iter=self.count_iter)

    n_out = self.n_out or int(input_shape[-1])
    self.l_f = Linear('last', n_out, count_iter=self.count_iter)

  def call(self, inputs):
    # relu with low regularity
    x = tf.nn.relu(self.l_1(inputs))
    x = tf.nn.relu(self.l_2(x))

    # tanh with high regularity
    #x = tf.nn.tanh(self.l_1(inputs))
    #x = tf.nn.tanh(self.l_2(x))

    x = self.l_f(x)

    return x


class beta_VAE_encoder(layers.Layer):
    def __init__(self,  name, n_dim,   # number of dimensions
                 #n_depth, # number of hidden layers.
                 n_width,  # number of neurons for each hidden layer
                 **kwargs):
        super(beta_VAE_encoder, self).__init__(name=name, **kwargs)

        self.n_dim = n_dim
        #self.n_depth = n_depth
        self.n_width = n_width

        self.encoder = NN2('encoder', n_width=n_width,
                           n_out=2*n_dim)
    
    
    def call(self, inputs):
        x = inputs
        z = self.encoder(x)
        mean_en = z[:,:self.n_dim]
        #std = 1e-6 + tf.nn.softplus(x[:, self.n_dim:])
        std_en = tf.exp(z[:, self.n_dim:])
               
        return mean_en, std_en

class beta_VAE_decoder(layers.Layer):
    def __init__(self, name,
                 data_dim,   # number of dimensions
                 #n_depth, # number of hidden layers.
                 n_width,  # number of neurons for each hidden layer
                 **kwargs):
        super(beta_VAE_decoder, self).__init__(name=name, **kwargs)

        self.data_dim = data_dim
        #self.n_depth = n_depth
        self.n_width = n_width

        self.decoder = NN2('decoder', n_width=n_width,
                           n_out=2*data_dim)

    def call(self, inputs):
        z = inputs
        x = self.decoder(z)
        mean = x[:,:self.data_dim]
        #std = 1e-6 + tf.nn.softplus(y[:, self.n_dim:])
        std = tf.exp(x[:, self.data_dim:])
        return mean, std



        
  
     






