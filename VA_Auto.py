# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:43:02 2017

@author: y_moh
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder(object):
      def __init__(self, hidden_1_n , hidden_2_n , hidden_3_n , latent_n):
          self.hidden_1_n = hidden_1_n
          self.hidden_2_n = hidden_2_n
          self.hidden_3_n = hidden_3_n          
          self.latent_n =latent_n
    
          
      def encoder(self,X):
          self.X=X
          with tf.variable_scope('encoder1'):
               W = tf.get_variable("weight",[X.shape[1].value, self.hidden_1_n], initializer = tf.random_normal_initializer(stddev=0.001))
               b = tf.get_variable("bias",[1, self.hidden_1_n], initializer = tf.zeros_initializer())
               layer_1 = tf.nn.relu(tf.add(tf.matmul(X,W),b))
               
          with tf.variable_scope('encoder2'):
               W = tf.get_variable("weight",[self.hidden_1_n, self.hidden_2_n], initializer = tf.random_normal_initializer(stddev= 1. / np.sqrt(self.hidden_1_n / 2.)))
               b = tf.get_variable("bias",[1, self.hidden_2_n], initializer = tf.zeros_initializer())
               layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,W),b))
               
          with tf.variable_scope('encoder3'):
               W = tf.get_variable("weight",[self.hidden_2_n, self.hidden_3_n], initializer = tf.random_normal_initializer(stddev= 1. / np.sqrt(self.hidden_2_n / 2.)))
               b = tf.get_variable("bias",[1, self.hidden_3_n], initializer = tf.zeros_initializer())
               layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2,W),b))
#               
          with tf.variable_scope('mu'):
               W = tf.get_variable("weight",[self.hidden_3_n, self.latent_n], initializer = tf.random_normal_initializer(stddev = 0.001))
               b = tf.get_variable("bias",[1, self.latent_n], initializer = tf.zeros_initializer())
               mu = tf.add(tf.matmul(layer_3,W),b)   
               
          with tf.variable_scope('std'):
               W = tf.get_variable("weight",[self.hidden_3_n, self.latent_n], initializer = tf.random_normal_initializer(stddev = 0.001))
               b = tf.get_variable("bias",[1, self.latent_n], initializer = tf.zeros_initializer())
               std = tf.add(tf.matmul(layer_3,W),b)
               
          return mu, std      
      
          
      def decoder(self,Y): 

          with tf.variable_scope('decoder1',reuse=None):
               W = tf.get_variable("weight",[self.latent_n, self.hidden_3_n], initializer = tf.random_normal_initializer(stddev=0.001))
               b = tf.get_variable("bias",[1, self.hidden_3_n], initializer = tf.zeros_initializer())
               layer_1 = tf.nn.relu(tf.add(tf.matmul(Y,W),b))
               
          with tf.variable_scope('decoder2',reuse=None):
               W = tf.get_variable("weight",[self.hidden_3_n, self.hidden_2_n], initializer = tf.random_normal_initializer(stddev=0.001))
               b = tf.get_variable("bias",[1, self.hidden_2_n], initializer = tf.zeros_initializer())
               layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,W),b))
               
          with tf.variable_scope('decoder3',reuse=None):
               W = tf.get_variable("weight",[self.hidden_2_n, self.hidden_1_n], initializer = tf.random_normal_initializer(stddev=0.001))
               b = tf.get_variable("bias",[1, self.hidden_1_n], initializer = tf.zeros_initializer())
               layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2,W),b))      
                              
              
          with tf.variable_scope('decoder',reuse=None):
               W = tf.get_variable("weight",[self.hidden_1_n, 6400], initializer = tf.random_normal_initializer(stddev=0.001))  
               b = tf.get_variable("bias",[1, 6400], initializer = tf.zeros_initializer())
               layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3,W ),b))
               return layer_4

          
