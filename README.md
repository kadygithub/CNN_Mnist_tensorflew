# Implementation of Convolutional Neural Network for MNIST Data Set with Tensorflow

In this project, we construct and implement Convolutional Neural Networks in Python with the TensorFlow framework. Multiple techniques, such as dropout, batch normalization, tuning hyperparameters using evaluation of the training-validation loss and accuracy between epochs  were implemented to enhance the performance of the model.

## Network Architecture

Many different CNN architectures for classification of Mnist data with high accuracy can be implemented. After running several experiments, I chose a model to find the architecture with the highest accuracy  and less computational complexity. The CNN with 4 convolutional layers,  two max-pooling layers (implemented by a convolutional layers), and one fully-connected layer  as follows:

  - Input layer: 784 nodes (MNIST images size)
  - First convolution layer: 3x3x32
  - Second convolution layer: 3x3x32
  - First max-pooling layer implemented as a convolutional layer with stride 2: 5x5x64
  - Third convolution layer : 3x3x64
  - Forth convolution layer : 3x3x64
  - Second max-pooling layer implemented as a convolutional layer with stride 2: 5x5x64
  - First fully-connected layer: 128 nodes 
  - Output layer: 10 nodes (number of classes for MNIST data)

  
## Techniques for improving performance and reliability

  - ### Introducing more non-linearity to the model
   By replacing one 5x5 convolution layer with two consecutive 3x3 layers, as well as adding a 5x5 convolution layer with strides=2 instead of max-pooling layer for subsampling. This technique benefits the performance since it is learnable.
  - ### Batch normalization 
 batch normalization is applied after all convolution/fully-connected layers. There is a debate if to apply batch normalization after or before activation function. In this work, I applied after activation.
 - ### Ensemble model
 In order to reducing the variance of the network , I train multiple models instead of a single model and to combine the predictions from the top 5 models in terms of performance. This method reduces the variance of predictions and results in predictions that perform better than any single model.
   
### Hyperparameters Tuning
   - ### Learning rate
  First, I use fixed learning rate=0.001 and Adam optimization which has an adaptive learning rate. The adam optimization algorithm in TensorFlow uses this value for the learning rate (alpha) based on the recommendation of Adam paper: learning_rate=0.001. However, I conducted the following randomized search to have a good choice of optimizer and learning rate:

   ![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/lr.png "fixed learning late with Adam optimization"). considering some values for learning rate, we measure in what time and in how many iterations (epochs), the training model gets to the accuracy of 98% or up.The one with the minimum training time is an efficient learning rate for Adam optimization (learning rate=.001)
    
   second, I  apply exponential decay to the learning rate which can also be used with Adam optimization.
   after randomized search, the following parameters were used for the exponential decay : the base learning rate:0.001 and decay rate is set to 0.95.
   ![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/decay_lr.png "exponential learning late with Adam optimization"). 
  comparing the first and second techniques for tuning the learning rate reveals that the fixed learning rate used with Adam optimizer perform better in terms of training speed.

   
### Dropout for Regularization
 In order to avoid overfitting of the model, after each max-pooling layers (conv layer  with stride 2) and the fully-connected layer dropouts are applied. We run multiple experiments to determine how much dropout should be considered after each layer. the following plots show how applying dropout give the model more of an opportunity to learn independent representations.
  ![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/NoDropout.png "no dropout")
 #### No dropout was applied. The validation accuracy increased and then decreased after 50-60 epochs, but then started to increase. The performance on the train set is good and almost becomes stationary distribution, whereas performance on the validation set improved to a point (within circle) and then began to degrade.This is a sign of overfitting such that the model tried to memorize the data. 
 
 
  ![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/70.png "Dropout rate :30%")
  Randomly 30% of the neurons were selected and set their weights to zero for the forward and backward passes i.e. for one epoch.
  The training loss approaches zero, but after some point (25-30 epochs) the validation loss increases. this is the sign of  overfitting again.
 
 ![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/40drop.png "Dropout rate :40%")
  Randomly 40% of the neurons were selected and set their weights to zero for the forward and backward passes.
 
 ![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/50.png "Dropout rate :50%")

  Randomly 50% of the neurons were selected and set their weights to zero for the forward and backward passes.

![Alt text](https://github.com/kadygithub/CNN_Mnist_tensorflew/blob/master/data/40.png "Dropout rate :60%")
 Randomly 60% of the neurons were selected and set their weights to zero for the forward and backward passes.

considering the above figures, it seems 50% dropout avoid overfitting while preserve high performance and stable convergence.

  
### Optimal number of iterations (epochs) 
The plots above, gives a hint about the value of optimal epoch. Considering dropout rate of 50%, we can see the optimization reaches some local minima and continuous to stay around that minima with stationary distribution. Therefore, the optimal value should be around a point that the local optimum was reached. However, number of epochs should be increased with a small batch size. I consider number of epochs=34.

### Batch size
After several experiments with different batch sizes, it seems 128 to be a good choice.

## Getting Started
This project was implemented at Colab Notebooks. It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud. We can upload the notebook to a GitHub repository or download.py files directly.
 link: https://colab.research.google.com/notebooks

### Software requirements
 - Tensorflow 1.13.1
 - Python 2.7 or Python 3
 - Numpy version 1.16.3

### The notbook version of this project in Github that can be uploaded directly to colab and easy to run:
https://github.com/kadygithub/Colab_MNIST_Akram_Tensorflow/blob/master/colab_MNIST_AKRAM.ipynb


### MNIST Dataset
Link: http://yann.lecun.com/exdb/mnist/

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. 5000 images out of 60,000 training images were selected for validation and the remaining images (55,000 images) were used for training. The digits have been size-normalized and centered in a fixed-size image 28*28.

Data_Mnist.py downloads the data to the folder named "data" and then pulls the images into a 4D tensor [image_index, y, x, channels] and then flatten out the later 3 dimensions [image_index, y*x*channels]: [image_index, 28*28*1]


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

The CNN model was trained and evaluated 100 times and the top five models in terms of accuracy were selected and stored in "model" directory. By running the test_Mnist_cnn.py, these five models are combined by majority voting technique. 

```
for instance: python test_Mnist_cnn.py --model-dir model --batch-size 5000 --is-ensemble True 
In colab : !python test_Mnist_cnn.py --model-dir model --batch-size 5000 --is-ensemble True
```
<model_directory> is the location of directory containing the sub-directories. Each sub-directories contains a saved model

## Simulation Results

The CNN network with the unique hyper-parameters has been trained 34 times, with fixed learning rate of=0.001 and adam optimization, batch size=128, dropout=50%.

- ### With 34 iterations and 50 % dropout, we get accuracy of 99.15% .
  (the model is saved in "model/single_model".)
 
- ###  Accuracy for the ensemble models: 99.49% 
 (the models are saved in "model/ensemble_models".)


## Acknowledgments

The implementation has been tested on Google colab :https://colab.research.google.com
The not3book version of this project in Github: https://github.com/kadygithub/Colab_MNIST_Akram_Tensorflow/blob/master/colab_MNIST_AKRAM.ipynb 
