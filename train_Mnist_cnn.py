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

import Data_Mnist
import model_cnn

# -------------------------------------------------   Training Phase
save_model_path = "model/model.ckpt"

# Hyper parameters for train
epochs = 200 # number of training itteration
TRAIN_BATCH_SIZE = 128 # batch size in traning phase
keep_probability = 0.7 # dropout percentage
learning_rate = 0.001  # learning rate

# Hyper parameters for test
TEST_BATCH_SIZE = 5000  #batch size for test phase
 
def train():
  # load data from Data_Mnist class
  d=Data_Mnist()
  train_data,train_labels,train_total_data, train_size, validation_data, validation_labels, test_data, test_labels=d.Produce_data(data_augmentation_Flag=False)
  # The image is initially loaded as a 784-dimensional vector. reshape the 784-dimensional vector to a 28 x 28 x 1 matrix. 
  train_X = train_data.reshape(-1, 28, 28, 1)
  test_X = test_data.reshape(-1,28,28,1)
  validation_X = validation_data.reshape(-1,28,28,1)
  print("validation_data",validation_data.shape)
  print("validation_labels",validation_labels.shape)
  train_y = train_labels
  test_y = test_labels
  validation_y = validation_labels

  # Define an input placeholder x, which will have a dimension of None x 784. None refers to batch size when we feed in the data.
  # Define an input placeholder y, which will have a dimension of None x 10 and holds the label of the training images
  x = tf.placeholder("float", [None, 28,28,1])
  y = tf.placeholder("float", [None, 10]) 
  # THE probabilities for each class label by calling cnn_net function and passing input image x
  pred = cnn_net(x)
  # The loss function is cross entropy. Both the activation (softmax) and the cross entropy loss functions are defined in one line. 
  # The predicted output and the ground truth label y are passed as input
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
  # Apply optimization algorithms: the Adam optimizer and specify the learning rate
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
  # Check whether the index of the maximum value of the predicted image is equal to the actual labelled image. 
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

  # Calculate accuracy over all the images and average them out. 
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # The Saver class adds ops to save and restore variables in order to initialize weights and biases
  saver = tf.train.Saver()
  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()
  # Lunch the graph in a session to run all the TensorFlow operations.
  with tf.Session() as sess:
    sess.run(init) 
    # initialize lists to keep the loss and accuracy for train, validation and test for each batch
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    validation_accuracy=[]
    validation_loss=[]
    # max_accuracy variable holds a value of maximum accuracy and update the model if the validation accuracy is greater than it.
    max_accuracy = 0.
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    # Loop over number of epochs
    for i in range(epochs):
        # Loop over number of batchs in each epoch : number of images/batch size
        for batch in range(len(train_X)//TRAIN_BATCH_SIZE):
            batch_x = train_X[batch*TRAIN_BATCH_SIZE:min((batch+1)*TRAIN_BATCH_SIZE,len(train_X))]
            batch_y = train_y[batch*TRAIN_BATCH_SIZE:min((batch+1)*TRAIN_BATCH_SIZE,len(train_y))]    
            # Run optimization 
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        

        
        # Calculate accuracy for all 10000 Mnist validation images
        
        validation_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: validation_X,y : validation_y})
        print("Epoch:", '%04d,' % (i + 1),"Validation Accuracy:","{:.5f}".format(validation_acc))
        # Update the model if the validation accuracy is greater max_acc
        if validation_acc > max_accuracy:
          max_accuracy = validation_acc
          save_path = saver.save(sess, save_model_path)
          print("Model updated and saved in file: %s" % save_path)
        
        test_acc1,test_loss1 = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        print("Epoch:", '%04d,' % (i + 1),"test Accuracy:","{:.5f}".format(test_acc1))
        
        # append the loss and accuracy to thier corresponding lists
        train_loss.append(loss)
        validation_loss.append(valid_loss)
        train_accuracy.append(acc)
        validation_accuracy.append(validation_acc)
        test_loss.append(test_loss1)
        test_accuracy.append(test_acc1)
        
    print("Adam Optimization process Finished!")  
    
    #---------------------------------   test the model
    
    # Restore variables 
    saver.restore(sess, save_model_path)

    # Calculate accuracy for all Mnist test images your the model is trained completely
    test_size = test_labels.shape[0]
    batch_size = TEST_BATCH_SIZE
    total_batch = int(test_size / batch_size)

    acc_buffer = []

    # Loop over all batches
    for i in range(total_batch):
        # Compute the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = test_X[offset:(offset + batch_size), :]
        batch_ys = test_y[offset:(offset + batch_size), :]
        # calculate the accuracy across all batchs
        y_final = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})
        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
        acc_buffer.append(numpy.sum(correct_prediction) / batch_size)
    # Plot the accuracy and loss plots between training and validation data:
    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), validation_loss, 'g', label='validation loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training, Validation and Test loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()   
    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_loss)), validation_accuracy, 'g', label='validation Accuracy')
    plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
    plt.title('Training, Validation and Test Accuracy')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
    summary_writer.close()

train()
