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

### Data loadig #########
images = load_data()


hidden_n_1 = 1000
hidden_n_2 = 500
hidden_n_3 = 200
latent_n = 20

X = tf.placeholder(tf.float32, [None, 6400] , "input")


A = Autoencoder(hidden_n_1 , hidden_n_2 , hidden_n_3 , latent_n)

mu, log_var = A.encoder(X)

unit_gaussian_samples = tf.random_normal([tf.shape(X)[0] , latent_n] , 0 , 1 , dtype=tf.float32)
latent_sample = mu + tf.exp(0.5 * log_var) * unit_gaussian_samples 
  
decoded = A.decoder(latent_sample)

latent_loss = 0.5 * tf.reduce_sum(tf.exp(log_var) + tf.square(mu) - 1. - log_var , axis=1)  

reconstruction_loss=tf.reduce_sum(tf.pow((decoded - X),2),1)

#decoded = tf.clip_by_value(decoded, 1e-8, 1 - 1e-8)
#reconstruction_loss=-tf.reduce_sum(X * tf.log(decoded) +(1 - X) * tf.log(1 - decoded), 1)

loss = tf.reduce_mean( latent_loss+reconstruction_loss)

op = tf.train.AdamOptimizer(0.0001).minimize(loss)


init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
init = tf.local_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:
     sess.run (init_op)
     sess.run (init)
     saver.restore(sess, "model/model8.ckpt")

     coord = tf.train.Coordinator()
     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
     for i in range(20001):         
         im=sess.run (images)
         #imm=imresize(im,[10,64,64,3])
         im2=im.astype(np.float32)/255
         im2=np.reshape(im2,[10,-1])
         _,L=sess.run([op,loss], feed_dict = {X: im2})
         print("iter " , i, "loss = ",L)
         if i%1000 ==0 :            
            save_path = saver.save(sess, "model/model9.ckpt")
     coord.request_stop()
     coord.join(threads)

         




      

    
