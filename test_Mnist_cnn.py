from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import ndimage
from six.moves import urllib
# user input
from argparse import ArgumentParser 

import numpy
import os
import tensorflow as tf
import gzip

import Data_Mnist
import model_cnn


# refernce argument values
MODEL_DIRECTORY = "model"
TEST_BATCH_SIZE = 5000
IS_ENSEMBLE = True

# parser to makes it easy to write user-friendly command-line interfaces.
def parser_interface():
    parser = ArgumentParser()

    parser.add_argument('--model-dir',
                        dest='model_directory', help='directory where model to be tested is stored',
                        metavar='MODEL_DIRECTORY', required=True)
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size for test',
                        metavar='TEST_BATCH_SIZE', required=True)
    parser.add_argument('--is-ensemble',
                        dest='ensemble', help='boolean for usage of ensemble',
                        metavar='IS_ENSEMBLE', required=True)
    return parser

def test_single(model_directory, batch_size):
    
    # load data from Data_Mnist class
    d=Data_Mnist.Data_Mnist()
    train_data,train_labels,train_total_data, train_size, validation_data, validation_labels, test_data,   test_labels=d.Produce_data(data_augmentation_Flag=False)
    test_X = test_data.reshape(-1,28,28,1)
    test_y = test_labels
    
    # tf Graph input
    x = tf.placeholder(tf.float32, [None,28,28,1])
    y_ = tf.placeholder(tf.float32, [None, 10])  
    y = model_cnn.cnn_net(x)
    

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk
    saver = tf.train.Saver()

    # Calculate accuracy for all mnist test images
    test_size = test_labels.shape[0]
    
    total_batch = int(test_size / batch_size)


    saver.restore(sess, model_directory)

    acc_buf = []

    # Loop over all batches
    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = test_X[offset:(offset + batch_size), :]
        batch_ys = test_y[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))

        acc_buf.append(numpy.sum(correct_prediction) / batch_size)

    print("Test accuracy for the stored model: %g" % numpy.mean(acc_buf))
    # For a given matrix, each row is converted into a one-hot row vector
def one_hot_matrix(a):
    a_ = numpy.zeros_like(a)
    for i, j in zip(numpy.arange(a.shape[0]), numpy.argmax(a, 1)): a_[i, j] = 1
    return a_

# test ensemble model (5 models) with test data 
def test_ensemble(model_directory_list, batch_size):
    # load data from Data_Mnist class
    d=Data_Mnist.Data_Mnist()
    train_data,train_labels,train_total_data, train_size, validation_data, validation_labels, test_data,   test_labels=d.Produce_data(data_augmentation_Flag=False)
    test_X = test_data.reshape(-1,28,28,1)
    test_y = test_labels
    
    # tf Graph input
    x = tf.placeholder(tf.float32, [None,28,28,1])
    y_ = tf.placeholder(tf.float32, [None, 10])  
    y = model_cnn.cnn_net(x)
    

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk
    saver = tf.train.Saver()

    # Calculate accuracy for all mnist test images
    test_size = test_labels.shape[0]
    
    total_batch = int(test_size / batch_size)
    acc_list = []

    # Loop over all batches
    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = test_X[offset:(offset + batch_size), :]
        batch_ys = test_y[offset:(offset + batch_size), :]

        y_final = numpy.zeros_like(batch_ys)
       
        for dir in model_directory_list:
            saver.restore(sess, dir+'/model.ckpt')
            pred = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
            # majority voting
            y_final += one_hot_matrix(pred) 
            print(y_final,"y_final..................................................................1")
        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
        print(correct_prediction,"correct prediction..................................................................2, out of loop")
        acc_list.append(numpy.sum(correct_prediction) / batch_size)
        print(acc_list,"sum of correct predictions..................................................................2, out of loop")
        
    print("Test accuracy for the ensemble models: %g" % numpy.mean(acc_list))
if __name__ == '__main__':
    # Parse arguments
    parser = parser_interface()
    options = parser.parse_args()
    ensemble = options.ensemble
    model_directory = options.model_directory
    batch_size = options.batch_size
   
    # Select a single model test or ensemble model
    if ensemble=='True': 
        # use ensemble model
        model_directory_list = [x[0] for x in os.walk(model_directory)]
        test_ensemble(model_directory_list[1:], batch_size)
    else: 
        # test a single model
        test_single(model_directory,batch_size)  
