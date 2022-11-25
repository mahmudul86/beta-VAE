from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import time
import beta_vae_layers as beta_vae_layers
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model

#the first model of our framework, the generator model
class beta_VAE(tf.keras.Model):
    def __init__(self, name,
                 data_dim, n_dim,  
                 n_width,
                 batch_size,
                 beta,
                 **kwargs):
        super(beta_VAE, self).__init__(name=name, **kwargs)

        self.data_dim = data_dim
        self.n_dim = n_dim
        self.n_width = n_width
        self.batch_size=batch_size
        self.beta=beta
        
        self.encoder_network = beta_vae_layers.beta_VAE_encoder('encoder_etwork', self.n_dim, self.n_width)
        self.decoder_network=beta_vae_layers.beta_VAE_decoder('decoder_network',self.data_dim, self.n_width)


    def call(self, inputs):
        #receives the combined input, then separates the ground truth part from the discriminator parameter values
        x = inputs
        x=tf.reshape(x,[self.batch_size,self.data_dim])
        
        #the encoder operation is done on the ground truth data
        #encoder part of the generator cost function calculated
        #samples for the latent space generated
        #1e-20 been added to avoid nan/inf. 
        mean_en,std_en=self.encoder_network(x)
        KL_divergence= -0.5*(tf.math.log(tf.math.square(std_en)+1e-20) + 1.0 - tf.math.square(mean_en) -tf.math.square(std_en))
        KL_divergence = tf.reduce_sum(KL_divergence, [1], keepdims=True)
        KL_divergence=tf.reduce_mean(KL_divergence)
        sample_z=mean_en+std_en*tf.random.normal((self.batch_size,self.n_dim))
        #the decoder operation is done on the samples generated at the encoder output
        #decoder part of the cost function calculated and added with the encoder part
        mean_de, std_de= self.decoder_network(sample_z)
        #print('std de', std_de)
        log_pdf= -0.5*(tf.math.square(x - mean_de)/ tf.math.square(std_de) + tf.math.log(2.0 * np.pi) + tf.math.log(tf.math.square(std_de)+1e-20))
        log_pdf= tf.reduce_sum(log_pdf, [1], keepdims=True)
        log_pdf=tf.reduce_mean(log_pdf)

        beta_vae_cost=self.beta*KL_divergence - log_pdf
        return beta_vae_cost, KL_divergence, log_pdf

