# Implementation of Convolutional Neural Network for MNIST Data Set with Tensorflow

In this project, we construct and implement Convolutional Neural Networks in Python with the TensorFlow framework. Multiple techniques, such as dropout, batchnormalization, evaluation of the training-validation loss between epochs.

## Network architecture

Many CNN architectures for classification of Mnist data with high accuracy can be implemented.We choose a model after running several experiments to find the architecture with the most accuracy  and efficiency (less computational complexity). The CNN with 4 convolutional layers and two max-pooling layers (implemented by a convolutional layers) and one fully-connected layer has following architecture:

  
  - input layer : 784 nodes (MNIST images size)
  - first convolution layer : 3x3x32
  - second convolution layer : 3x3x32
  - first max-pooling layer implemented as a convolutional layer with stride 2: 5x5x64
  - third convolution layer : 3x3x64
  - forth convolution layer : 3x3x64
  - second max-pooling layer implemented as a convolutional layer with stride 2: 5x5x64
  - first fully-connected layer : 128 nodes
  - output layer : 10 nodes (number of classes for MNIST data)  
### Techniques for improving performance and reliabilty
  - ### Adding more nonlinearity 
  By replacing one 5x5 convolution layer with two consecutive 3x3 layers, Furthermore, adding a 5x5 convolution layer with strides=2 instead of max-pooling layer for subsampling that benefit the performance since it is learnable.
  - ### Batch normalization 
  All convolution/fully-connected layers use batch normalization.
  - ### Dropout 
  After each max-pooling layers and the fully-connected layer dropout technique is added in order to reduce the overfitting of the      model. We run experiment multiple times to determine how much dropout should be considered after each layer. The results shows 40% dropout gives the best results.
  - ### Optimal number of iterations (epochs) 
  After evaluation of the training-validation loss and accuracy between epochs, we consider epochs=34. 
  

## Getting Started
This project was implemented at Colab Notebooks. It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud. We can upload the notebook to a GitHub repository or download.py files directly.
 

### MNIST Dataset
Link: http://yann.lecun.com/exdb/mnist/

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image 28by28. Data_Mnist.py download the data to the folder named "data" and then pulls the images into a 4D tensor [image_index, y, x, channels] nad then flatten out the later 3 dimensions [image_index, y*x*channels]:[image_index, 28*28*1]


### Train

python train_Mnist_cnn.py

The trained model is saved as "model/model.ckpt".

### Test

python test_Mnist_cnn.py --model-dir <model_directory> --batch-size <batch_size> 

<model_directory> is the location where a model to be testes is saved without specifying filename of "model.ckpt".
<batch_size> is the number of training examples utilized in one iteration of test data with 10,000 images. 
```
for instance: python test_Mnist_cnn.py --model-dir modeltest/model.ckpt --batch-size 5000 
In colab : !python test_Mnist_cnn.py --model-dir modeltest/model.ckpt --batch-size 5000 
```

## Simulation Results

The CNN network with the unique hyper-parameters has been trained 200 times, and after evaluating the loos and accuracy, the number of epochs has been chosed to be: 34. The reason is that the loss start to increase after 30-35 epochs while accuracy dosn't change considerably.
![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/70.png "Dropout :70%")

![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/40.png "Dropout :40%")
With 34 iterations and dropout of 0.7, we get accurcy of 
- ## 99.61% of accuracy 99.45% .
 (the model is saved in "model/single_model".)






## Acknowledgments

The implementation has been tested on Google colab :https://colab.research.google.com
The notbook version of this project in Github: 
