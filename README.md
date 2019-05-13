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
  1- Batch normalization
   All convolution/fully-connected layers use batch normalization.
  2- Dropout
  After each max-pooling layers and the fully-connected layer dropout technique is added in order to reduce the overfitting of the      model. We run experiment multiple times to determine how much dropout should be considered after each layer. The results shows 40% dropout gives the best results.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

