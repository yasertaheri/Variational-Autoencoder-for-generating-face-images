# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 00:17:20 2017

@author: y_moh
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from VA_Auto import Autoencoder
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

Batch_size=100
tf.reset_default_graph()

hidden_n_1 = 500
hidden_n_2 = 500
hidden_n_3 = 200
latent_n = 2

X = tf.placeholder(tf.float32, [None, 784] , "input")

A = Autoencoder(hidden_n_1 , hidden_n_2 , hidden_n_3 , latent_n)

mu, log_var = A.encoder(X)

unit_gaussian_samples = tf.random_normal([Batch_size , latent_n] , 0 , 1 , dtype=tf.float32)
latent_sample = mu + tf.exp(0.5 * log_var) * unit_gaussian_samples 
  
decoded = A.decoder(latent_sample)

latent_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + tf.square(mu) - 1. - log_var , axis=1)  

#reconstruction_loss=tf.reduce_sum(tf.pow((decoded - X),2),1)

decoded = tf.clip_by_value(decoded, 1e-8, 1 - 1e-8)
reconstruction_loss=-tf.reduce_sum(X * tf.log(decoded) +(1 - X) * tf.log(1 - decoded), 1)

loss = tf.reduce_mean( latent_loss+reconstruction_loss)

op = tf.train.AdamOptimizer(0.001).minimize(loss)


init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
     sess.run(init)
     for i in range(50000):
         
         _,L=sess.run([op,loss], feed_dict = {X: mnist.train.next_batch(Batch_size)[0]})
         print("iter " , i, "loss = ",L)
         if i%1000 ==0 :            
            save_path = saver.save(sess, "model/model.ckpt")
    

##### test#########

      

    
