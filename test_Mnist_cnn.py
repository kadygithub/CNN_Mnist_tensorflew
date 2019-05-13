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


# build parser
def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--model-dir',
                        dest='model_directory', help='directory  of the model to be tested',
                        metavar='MODEL_DIRECTORY', required=True)
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size for test',
                        metavar='TEST_BATCH_SIZE', required=True)
    
    return parser

def test_org(model_directory, batch_size):
    
    # load data from Data_Mnist class
    d=Data_Mnist()
    train_data,train_labels,train_total_data, train_size, validation_data, validation_labels, test_data, test_labels=d.Produce_data(data_augmentation_Flag=False)
    # Import data
    
    

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])  
    y = cnn_net(x)
    

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
        batch_xs = test_data[offset:(offset + batch_size), :]
        batch_ys = test_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))

        acc_buf.append(numpy.sum(correct_prediction) / batch_size)

    print("Test accuracy for the stored model: %g" % numpy.mean(acc_buf))
if __name__ == '__main__':
    # Parse argument
    parser = build_parser()
    options = parser.parse_args()
    ensemble = options.ensemble
    model_directory = options.model_directory
    batch_size = options.batch_size
    test(model_directory+'/model.ckpt',batch_size)  
