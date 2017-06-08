# TensorFlow-Slim 

This README is a summary of this README aboout TF-slim : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim


TF-Slim is a lightweight library for defining, training and avaluating complex models in TensorFlow. 
Components of TF-Slim can be freely miwed with native tensorflow, as well as other frameworks such as tf.contrib.learn 

## Usage 

```python
import tensorflow.contrib.slim as slim 
```

## Why TF-Slim? 

TF-Slim is a library that makes building, training, evaluation neural network simple: 

* Allows the user to define models more compactly by eliminating boilerplate code. This is accomplished through the use of   **argument scoping**, **high level layers** and **variables**. This tools increase the readability maintainability, reduce the likelihood of an error from copy and pasting the hyperparameter values ans simplifies hyperparameters tuning. 

* Makes developping models simple by providing commonly used **regularizer** 

* Several widely used computer vision models have been developped in slim and are available to the public  : [Alexnet](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py),  [Inception](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception.py), [Resnet](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py), [VGG](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg_test.py)

## What are the various components of TF-Slim? 

TF-Slim is composed of several parts which were design to exist independently: arg_scope, data, evaluation, layers, learning, losses, metrics, nets, queues, regularizers, variables. 

## Defining Models 

Models can be defined by using variables, layers and scopes. 

### Variables 
Creation of a ```weight```variable : initialize it using a truncated normal distribution, regularize it with an ```l2_loss```and place it on the ``` CPU ```. With TF-Slim : model variables are  variables that represent parameters of the model. 

```python
weights=slim.model_variable('weights', 
shape [10,10,3,3], 
initializer=tf.truncated_normal_initializer(stddev=0.1), 
regularizer=slim.l2_regularizer(0.05), 
device='/CPU:0')
model_variables= slim.get_model_variable()
```


### Layers 

### Scopes 
