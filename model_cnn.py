



from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf


# Helper libraries
import numpy
import matplotlib.pyplot as plt
from scipy import ndimage
from six.moves import urllib

import gzip
import os

# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/kadygithub/8e9fb26697add316c661be1eb13a5ca0/untitled1.ipynb
"""
#-------------    create CNN structure by Akram ---------------------------------------------------
"""
A basic CNN structure for MNIST data classidication is two convolution layers followed by subsambling layers (e.g., maxpooling).
Then, classification is performed with a fully connected layer followed by a final softmax layer. For image classification,
this architecture performs better fairly good, however in this implementation, CNN structure is as follows:

Two convolution layer with 32 feature maps using a 3x3 filter and stride 1 (instead of one convolution with a 5x5 filter to add more none-linearity)
A convolution layer with 32 feature maps using a 5x5 filter and stride 2 (instead of max pooling)
70% dropout is added after subsampling layer in order to prevent overfitting
tion layer with 64 feature maps using a 3x3 filter and stride 1 (instead of one convolution with a 5x5 filter to add more none-linearity)
A convolution layer with 64 feature maps using a 5x5 filter and stride 2 (instead of max pooling)
70% dropout is added after subsampling layer in order to prevent overfitting
A fully connected layer with 128 units
A softmax layer
"""

# Define convolution function using Conv2D wrapper, bias and relu activation
def conv2d(x, W, b, strides,padding):
    # Conv2D wrapper with bias and relu activation
    # For conv2d, default argument values are like
    # input=x
    # filter=W
    # strides=[1,strides,strides,1] strides[0] & strides[3] must always be 1, because the first is for the image-number and the last is for the input-channel 
    # padding ='A string from: "SAME", "VALID". The type of padding algorithm to use.
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

# Create CNN stucture
def cnn_net(x):
  # Create a dictionery for weight parameters 
  # Each key (e.g., conv1_filter) in the weights dictionary has an argument shape that takes a tuple with 4 values:
  # The first and second are the filter size,  the third is the number of channels in the input image and the last is the number of convolution filters.
  # The key 'wfc' we flatten the output of last convolutional layer to feed this as input to the fully connected layer. 
  # so, we do the multiplication operation 4*4*64
  weights={
    'conv1_filter': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'conv2_filter': tf.get_variable('W1', shape=(3,3,32,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'conv3_filter': tf.get_variable('W2', shape=(5,5,32,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'conv4_filter': tf.get_variable('W3', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'conv5_filter': tf.get_variable('W4', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'conv6_filter': tf.get_variable('W5', shape=(5,5,64,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wfc': tf.get_variable('W6', shape=(4*4*64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W7', shape=(128,10), initializer=tf.contrib.layers.xavier_initializer()), 
  }
  # Create a dictionery for bias parameters 
  # Each key (e.g., bc1) in biases dictionary has 32 bias parameters.
  # The bfc key has 128 parameters, the number of neurons in the fully connected layer.
  biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc5': tf.get_variable('B4', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc6': tf.get_variable('B5', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bfc': tf.get_variable('B6', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B7', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}
  # input [28,28,1] ---> [26,26,32]  
  conv1 = conv2d(x, weights['conv1_filter'], biases['bc1'],1,'VALID')
  conv1_bn = tf.layers.batch_normalization(conv1)
  # input [26,26,32] -----> [24,24,32] 
  conv2 = conv2d(conv1_bn, weights['conv2_filter'], biases['bc2'],1,'VALID')
  # add batch normalization
  conv2_bn = tf.layers.batch_normalization(conv2)
  # input [24,24,32] ---> [12,12,32] 
  conv3 = conv2d(conv2_bn, weights['conv3_filter'], biases['bc3'],2,'SAME')
  # add batch normalization
  conv3_bn = tf.layers.batch_normalization(conv3)
  dropl = tf.nn.dropout(conv3_bn, 0.7)
  # input [12,12,32] ---> [10,10,64]   
  conv4 = conv2d(dropl, weights['conv4_filter'], biases['bc4'],1,'VALID')
  conv4_bn = tf.layers.batch_normalization(conv4)
  # input [10,10,64] ---> [8,8,64] 
  conv5 = conv2d(conv4_bn, weights['conv5_filter'], biases['bc5'],1,'VALID')
  # add batch normalization
  conv5_bn = tf.layers.batch_normalization(conv5)
  # input [8,8,64] ---> [4,4,64] 
  conv6 = conv2d(conv5_bn, weights['conv6_filter'], biases['bc6'],2,'SAME')
  # add batch normalization
  conv6_bn = tf.layers.batch_normalization(conv6)
  # 70% dropout is added after subsampling layer in order to prevent overfitting
  drop2 = tf.nn.dropout(conv6_bn, 0.7)  
    
  # Fully connected layer

  # Reshape conv2 output to fit fully connected layer input  input [4,4,64] ---> [4*4*64,128] 
  fc1 = tf.reshape(drop2, [-1, weights['wfc'].get_shape().as_list()[0]])
  fc1 = tf.add(tf.matmul(fc1, weights['wfc']), biases['bfc'])
  fc1 = tf.nn.relu(fc1)
  # add batch normalization
  fc1_bn = tf.layers.batch_normalization(fc1)
  # 70% dropout is added after subsampling layer in order to prevent overfitting
  drop_fc = tf.nn.dropout(fc1_bn, 0.7)
  # Output: class prediction
  # At the end, we multiply the fully connected layer with the weights and add a bias term. 
  out = tf.add(tf.matmul(drop_fc, weights['out']), biases['out'])
  return out

