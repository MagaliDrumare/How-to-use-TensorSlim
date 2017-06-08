# TensorFlow-Slim 

TF-Slim is a lightweight library for defining, training and avaluating complex models in TensorFlow. 
Components of TF-Slim can be freely miwed with native tensorflow, as well as other frameworks such as tf.contrib.learn 

## Usage 

```
import tensorflow.contrib.slim as slim 
```

## Why TF-Slim? 

TF-Slim is a library that makes building, training, evaluation neural network simple: 

* Allows the user to define models more compactly by eliminating boilerplate code. This is accomplished through the use of** argument scoping**, **high level layers** and **variables**. This tools increase the readability maintainability, reduce the likelihood of an error from copy and pasting the hyperparameter values ans simplifies hyperparameters tuning. 

* Makes developping models simple by providing commonly used **regularizer** 

* Several widely used computer vision models have been developped in slim (VGG,AlexNet)and are available to the public  
    1.[Alexnet](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py)

