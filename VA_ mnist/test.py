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
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
tf.reset_default_graph()

hidden_n_1 = 500
hidden_n_2 = 500
hidden_n_3 = 200
latent_n = 2

A = Autoencoder(hidden_n_1 ,hidden_n_2,hidden_n_3, latent_n)
sample = tf.random_normal([1 , latent_n] , 0 , 1 , dtype=tf.float32)
c=A.decoder(sample)
saver = tf.train.Saver()
with tf.Session() as sess:
     saver.restore(sess, "model/model.ckpt")
     samples = tf.random_normal([1 , latent_n] , 0 , 1 , dtype=tf.float32)
     s=sess.run(samples)

     image = mnist.train.next_batch(1)[0]
     d=sess.run(c)
     plt.figure(1)
     plt.imshow(np.reshape(d, [28,28]), cmap="gray")
