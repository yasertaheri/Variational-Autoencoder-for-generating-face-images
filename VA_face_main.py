# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 00:17:20 2017

@author: Yaser M.Taheri
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from VA_Auto import Autoencoder
from load_data import load_data

Batch_size=10
tf.reset_default_graph()

## Data loadig from a dataset ###
images = load_data()

# Autoencoder shape [6400 1000 500 200 [20,20] 1000 500 200 6400]
# [20 20] stands for 20 nodes for mean vector and  20 nodes for log(variance) vector as latent variables

# number of nodes in the hidden layes 1,2,3 and latent layer
hidden_n_1 = 1000    #number of nodes in the 1st hidden layer
hidden_n_2 = 500     #number of nodes in the 2nd hidden layer
hidden_n_3 = 200     #number of nodes in the 3rd hidden layer
latent_n = 20        #number of nodes for the latent vactors


X = tf.placeholder(tf.float32, [None, 6400] , "input") # flatten grayscale input image of size [80,80] 


# Autoencoder object                   
A = Autoencoder(hidden_n_1 , hidden_n_2 , hidden_n_3 , latent_n)

# Encodig of variational Autoencoder with output of latent vectors that are mean vector (mu) 
# and log(variance) vector, log_var. 
mu, log_var = A.encoder(X)


# sample from normal distribution N
unit_gaussian_samples = tf.random_normal([tf.shape(X)[0] , latent_n] , 0 , 1 , dtype=tf.float32)

# sample from gaussian distribtion with mean of mu
latent_sample = mu + tf.exp(0.5 * log_var) * unit_gaussian_samples 
  
decoded = A.decoder(latent_sample)


# latent_loss : KL- divergece  for comparig the distribution of latent variables to a normal distribution N(0,1)
latent_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + tf.square(mu) - 1. - log_var , axis=1)   

# square loss by commparig the reconstructed image at output and iput image
reconstruction_loss=tf.reduce_sum(tf.pow((decoded - X),2),1)     


loss = tf.reduce_mean(latent_loss+reconstruction_loss)           # total loss

op = tf.train.AdamOptimizer(0.0001).minimize(loss)               # Adam optimizer with learnig rate of 0.0001


init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()) # initilizatio of all variables

saver = tf.train.Saver()


with tf.Session() as sess:
     sess.run (init_op)
     
     #saver.restore(sess, "model/model8.ckpt")

     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
     for i in range(20001):         
         im=sess.run (images)
         im2=im.astype(np.float32)/255
         im2=np.reshape(im2,[10,-1])
         _,L=sess.run([op,loss], feed_dict = {X: im2})
         print("iter " , i, "loss = ",L)
         if i%1000 ==0 :            
            save_path = saver.save(sess, "model/model9.ckpt")
     coord.request_stop()
     coord.join(threads)

         




      

    
