# TensorFlow-Slim 

This README is a summary of this README aboout TF-Slim written by Sergio Guadarrama and Nathan Siberman https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

# What is TensorFlow-Slim? 
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


## What are the various components of TF-Slim? 

TF-Slim is composed of several parts which were design to exist independently: arg_scope, data, evaluation, layers, learning, losses, metrics, nets, queues, regularizers, variables. 

## Defining Models 

Models can be defined by using variables, layers and scopes. 

### Variables 
Creation of a ```weight```variable : initialize it using a truncated normal distribution, regularize it with an ```l2_loss```and place it on the ``` CPU ```. New with TF-Slim : **model variables** which are  variables that represent parameters of the model. 

```python
weights=slim.model_variable('weights', 
shape [10,10,3,3], 
initializer=tf.truncated_normal_initializer(stddev=0.1), 
regularizer=slim.l2_regularizer(0.05), 
device='/CPU:0')
model_variables= slim.get_model_variable()
```

### Layers 

TF_Slim provides standard implementations for numerous components for building neral networks 

Layer	| Tf-Slim 
------|--------
BiasAdd |slim.bias_add
BatchNorm|slim.bias_norm 
Conv2d| slim.conv2d
FullyConnected | slim.fully_conected
AvgPool2D|slim.avg_pool2D
Dropout|slim.dropout
Flatten|slim.flatten
MaxPool2D|slim.max_pool2d
OneHotEncoding|slim.one_hot_encoding

TF_Slim also provides two meta-operation called ```repeat```and ```stack```that allow users to reapeatetly perform the same operation

* ```slim.repeat ```

```python

net=....
net=slim.conv2d(net, 256, [3,3], scope='conv3_1')
net=slim.conv2d(net, 256, [3,3], scope='conv3_2')
net=slim.conv2d(net, 256, [3,3], scope='conv3_2')
net=slim.max_pool_2d(net, [2,2], scope='pool2')

can be replaced by 

net=slim.repeat(net,3,slim.conv2d, 256,[3,3], scope='conv3')
net=slim.max_pool_2d(net, [2,2], scope='pool2']

```

* ```slim.stack ```

```python

x=slim.conv2d(x,32,[3,3], scope='core/core_1')
x=slim.conv2d(x,32,[1,1], scope='core/core_2')
x=slim.conv2d(x,64,[3,3], scope='core/core_3')
x=slim.conv2d(x,64,[1,1], scope='core/core_4')

can be replaced by 
x=slim.stack(x, slim.conv2d,[(32,[3,3]),(32,[1,1]),(64,[3,3]),(64,[1,1]), scope='core']

```

### Scopes 

TF_Slim adds a new scoping mechanism called ```arg_scope``` used to simplify the code. 

```python 
net=slim.conv2d(inputs, 64, [11,11],4, padding='SAME', 
weights_initializer=tf.truncated_normal_initializer(stddev=0.01), 
weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')

net=slim.conv2d(net, 128, [11,11],padding='VALID', 
weights_initializer=tf.truncated_normal_initializer(stddev=0.01), 
weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')

net=slim.conv2d(net,256, [11,11],padding='SAME', 
weights_initializer=tf.truncated_normal_initializer(stddev=0.01), 
weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')

can be replaced by 

with slim.arg_scope([slim.conv2d], padding='SAME', 
weights_initializer=tf.truncated_normal_initializer(stddev=0.01), 
weights_regularizer=slim.l2_regularizer(0.0005))

net=slim.conv2d(input, 64,[11,11], scope='conv1')
net=slim.conv2d(input, 128,[11,11],padding='VADID', scope='conv2')
net=slim.conv2d(input, 256,[11,11], scope='conv3')

``` 
Another example of code.the first ```arg_scope```applies ```weights_initializer``` and ```weights_regularizer``` to the ```conv2d``` and ```fully_connected``` layers in the scope. In the second ```arg scope ``` applies default arguments to 
```conv2d```. In the fully_coonected layer, the activation function is not activated : ```activation_fn=None ```. 

```python
with slim.arg_scope([slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu, 
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
      weights_regulizer=slim.l2.regularizer(0.00005)):
  with slim.arg_scope ([slim.conv2d, stride=1,padding='SAME'): 
  
  net=slim.conv2d(inputs, 64,[11,11], padding='VALID', scope='conv1')
  net=slim.con2d(net, 256,[5,5], weights_initializer(stddev=0.03), scope='conv2')
  net=slim.fully_connected(net, 100, activation_fn=None, scope='fc')
  
```






