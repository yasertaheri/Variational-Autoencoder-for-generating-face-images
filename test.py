# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:54:31 2017

@author: y_moh
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from VA_Auto import Autoencoder
tf.reset_default_graph()

hidden_n_1 = 1000
hidden_n_2 = 500
hidden_n_3 = 200
latent_n = 20

A = Autoencoder(hidden_n_1 ,hidden_n_2,hidden_n_3, latent_n)
sample = tf.random_normal([1 , latent_n] , 0 , 1 , dtype=tf.float32)
c=A.decoder(sample)
saver = tf.train.Saver()




with tf.Session() as sess:
     saver.restore(sess, "model/model8.ckpt")
     def plot():
        n = 9
        n_columns =3
        n_rows = np.ceil(n / n_columns) + 1
        for i in range(25):
            new_faces=sess.run(c)
            plt.figure(1)
            plt.subplot(n_rows, n_columns, i+1)
            plt.imshow(np.reshape(new_faces, [80,80]), cmap="gray")        
            plt.axis('off')

     plot()
         
     plt.axis('off')
