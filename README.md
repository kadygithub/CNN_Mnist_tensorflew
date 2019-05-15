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
## Techniques for improving performance and reliabilty
  - ### introducing more non-linearity to the model
   By replacing one 5x5 convolution layer with two consecutive 3x3 layers, Furthermore, adding a 5x5 convolution layer with strides=2 instead of max-pooling layer for subsampling that benefit the performance since it is learnable.
  - ### Batch normalization 
  All convolution/fully-connected layers use batch normalization. a
   
### Hyperparameters Tuning
   - ### Learning rate
   I first use fixed learning rate=0.001 and Adam optimization which has an adaptive learning rate. The adam optimization algorithm in 
   TensorFlow uses the following default values for parameters based on the recomendation of Adam paper.
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
    
   - second, I  apply exponential decay to the learning rate which can also be used with Adam optimization.
   after tandom search, the following papmeters were used for the exponential decay : the base learning rate:.001 and decay rate is set to 0.95.
  - comparing first and second techniques for tuning the learning rate reveals no significant improvment in the classification accuray.
   
   how much to update the weight in the optimization algorithm. We can use fixed learning rate, gradually decreasing learning rate, momentum based methods or adaptive learning rates, depending on our choice of optimizer such as SGD, Adam, Adagrad, AdaDelta or RMSProp.
 
  
  
  - ### Dropout for regularization
  After each max-pooling layers and the fully-connected layer dropout technique is added in order to reduce the overfitting of the      model. We run experiment multiple times to determine how much dropout should be considered after each layer. The results shows 40% dropout gives the best results.
  - ### Optimal number of iterations (epochs) 
  After evaluation of the training-validation loss and accuracy between epochs, we consider epochs=34. 
  

## Getting Started
This project was implemented at Colab Notebooks. It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud. We can upload the notebook to a GitHub repository or download.py files directly.
 link: https://colab.research.google.com/notebooks

### Software requirements
 - Tensorflow 1.13.1
 - Python 2.7 or Python 3
 - Numpy version 1.16.3
 

### MNIST Dataset
Link: http://yann.lecun.com/exdb/mnist/

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. 5000 images out of 60,000 training images were selected for validation and the remaining images (55,000 images) were used for training. The digits have been size-normalized and centered in a fixed-size image 28by28. Data_Mnist.py download the data to the folder named "data" and then pulls the images into a 4D tensor [image_index, y, x, channels] nad then flatten out the later 3 dimensions [image_index, y*x*channels]:[image_index, 28*28*1]


### Train

python train_Mnist_cnn.py

The trained model is saved in "model/single_model/model2.ckpt".

### Test

#### Test for Single model

python test_Mnist_cnn.py --model-dir <model_directory> --batch-size <batch_size> 

<model_directory> is the location where a model to be testes is saved without specifying filename of "model.ckpt".
<batch_size> is the number of training examples utilized in one iteration of test data with 10,000 images. 
```
for instance: python test_Mnist_cnn.py --model-dir model/model1/model.ckpt --batch-size 5000 --is-ensemble False
In colab : !python test_Mnist_cnn.py --model-dir model/model1/model.ckpt --batch-size 5000 --is-ensemble False
```
#### Test for Ensemble model
The CNN model was trained and evaluated 100 times and the top five models in terms of accuracy were selected and stored in "model" directory. By running the test_Mnist_cnn.py, these five models are ensembled by majority voting technique.

```
for instance: python test_Mnist_cnn.py --model-dir model --batch-size 5000 --is-ensemble True 
In colab : !python test_Mnist_cnn.py --model-dir model --batch-size 5000 --is-ensemble True
```
<model_directory> is the location of directory contaning the sub-directories. Each sub-directories contains a saved model

## Simulation Results

The CNN network with the unique hyper-parameters has been trained 200 times, and after evaluating the loos and accuracy, the number of epochs has been chosed to be: 34. The reason is that the loss start to increase after 30-35 epochs while accuracy dosn't change considerably.
![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/70.png "Dropout rate :30%")
 #### Randomly 30% of the neurons were selected and set their weights to zero for the forward and backward passes i.e. for one epoch.
 while :
 
![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/40.png "Dropout rate :60%")
 #### Randomly 60% of the neurons were selected and set their weights to zero for the forward and backward passes i.e. for one epoch.
 
- ### With 34 iterations and 30 % dropout, we get accurcy of 99.45% .
  (the model is saved in "model/single_model".)
 #### ----------------------------------------------------------  Dropout rate : 40% 
- ###  Accuracy for the ensemble models: 99.19%
- ###  Accuracy for the ensemble models: 99.49% 
 (the models are saved in "model/ensemble_models".)





## Acknowledgments

The implementation has been tested on Google colab :https://colab.research.google.com
The notbook version of this project in Github: 
