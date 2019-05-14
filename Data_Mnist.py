# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/kadygithub/8e9fb26697add316c661be1eb13a5ca0/untitled1.ipynb
"""



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

class Data_Mnist:
  def __init__(self):
    self.SOURCE = 'http://yann.lecun.com/exdb/mnist/'
    self.DATA_DIR= "data"
    self.IMAGE_SIZE = 28
    self.CHANNELS = 1
    self.INTENSITY = 255
    self.LABELS = 10
    self.filepath=""
    # number of the validation  data images.
    self.SIZE_VALIDATION= 5000  

# import data
  def import_data(self,filename):
      #check the DATA_DIR, if data dose not exist,make directory and then download data!
      if not tf.gfile.Exists(self.DATA_DIR):
          tf.gfile.MakeDirs(self.DATA_DIR)
      filepath = os.path.join(self.DATA_DIR, filename)
      if not tf.gfile.Exists(filepath):
          filepath, _ = urllib.request.urlretrieve(self.SOURCE + filename, filepath)
          print('data downloaded')
      
      return filepath
  def pull_images(self,filename, num_images):
    """First we pull the images into a 4D tensor [image_index, y, x, channels] 
      then flatten out the later 3 dimensions[image_index, y*x*channels]
    """
   
    print('Extracting', filename)
    with gzip.open(filename) as f:
        k=f.read(16)
        
        
        b = f.read(self.IMAGE_SIZE * self.IMAGE_SIZE * num_images * self.CHANNELS)
        # to use the memory buffer of the data directly
        data = numpy.frombuffer(b, dtype=numpy.uint8).astype(numpy.float32)
        #rescaling values of pixels : [0,255] --> [-.5,5]
        data = data/self.INTENSITY#(data - (self.INTENSITY)) /(2* self.INTENSITY)
        
        data = data.reshape(num_images, self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANNELS)
        
        # we don't need this layered information (num_images, 28, 28, 1), 
        #so we just flatten out the later 3 dimensions (num_images, 784) 
        data = numpy.reshape(data, [num_images, -1])
    
    return data
  
  def pull_labels(self,filename,num_images):
      # the labels are extracted into a vector"""
      with gzip.open(filename) as f:
          f.read(8)
          b = f.read(1 * num_images)
          labels = numpy.frombuffer(b, dtype=numpy.uint8).astype(numpy.int64)
          num_labels = len(labels)
          encoding= numpy.zeros((num_labels,self.LABELS))
          encoding[numpy.arange(num_labels),labels] = 1
          encoding = numpy.reshape(encoding, [-1, self.LABELS])
          
      return encoding
  def Produce_data(self,data_augmentation_Flag=False):
      # Download data
      train_data = self.import_data('train-images-idx3-ubyte.gz')
      train_labels =self.import_data('train-labels-idx1-ubyte.gz')
      test_data = self.import_data('t10k-images-idx3-ubyte.gz')
      test_labels =self.import_data('t10k-labels-idx1-ubyte.gz')

      # pull images and labels into numpy tensors
      train_data = self.pull_images(train_data,60000)
      print("Training set (images) shape: {shape}".format(shape=train_data.shape))
      train_labels = self.pull_labels(train_labels, 60000)
      print("Training set (labels) shape: {shape}".format(shape=train_labels.shape))
      test_data = self.pull_images(test_data, 10000)
      print("Test set (images) shape: {shape}".format(shape=test_data.shape))
      test_labels = self.pull_labels(test_labels, 10000)
      print("Test set (labels) shape: {shape}".format(shape=test_labels.shape))

      # Set validation data from training data
      validation_data = train_data[:self.SIZE_VALIDATION, :]
      validation_labels = train_labels[:self.SIZE_VALIDATION,:]
      train_data = train_data[self.SIZE_VALIDATION:, :]
      train_labels = train_labels[self.SIZE_VALIDATION:,:]

      #  data_concatenation
      train_data_Total = numpy.concatenate((train_data, train_labels), axis=1)

      train_size = train_data_Total.shape[0]

      return train_data,train_labels,train_data_Total, train_size, validation_data, validation_labels, test_data, test_labels
    

"""  
#d=Data_Mnist()
#train_data,train_labels,train_total_data, train_size, validation_data, validation_labels, test_data, test_labels=d.Produce_data(data_augmentation_Flag=False)
print("Training set (images) shape: {shape}".format(shape=train_data.shape))
print("Training set not include validation(images) shape: {shape}".format(shape=train_total_data.shape))
print( "the images are already rescaled between -0.5 and 0.5, max: ",  numpy.max(train_total_data[0]))
print( "the images are already rescaled between -0.5 and 0.5, min: ",  numpy.min(train_total_data[0]))
"""

