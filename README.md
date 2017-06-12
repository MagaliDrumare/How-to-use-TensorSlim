# TensorFlow-Slim 

This README is a summary of this README aboout TF-Slim written by Sergio Guadarrama and Nathan Siberman https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

# What is TensorFlow-Slim? 
TF-Slim is a lightweight library for defining, training and avaluating complex models in TensorFlow. 
Components of TF-Slim can be freely mixed with native tensorflow, as well as other frameworks such as tf.contrib.learn 

## Usage 

```python
import tensorflow.contrib.slim as slim 
```

## Why TF-Slim? 

TF-Slim is a library that makes building, training, evaluation neural network simple: 

* Allows the user to define models more compactly by eliminating boilerplate code. This is accomplished through the use of   **argument scoping**, **high level layers** and **variables**. This tools increase the readability and maintainability, reduce the likelihood of an error from copy and pasting the hyperparameter values ans simplifies hyperparameters tuning. 

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

### Layers, slim.repeat, slim.stack 

TF-Slim provides standard implementations for numerous components for building neral networks 

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

TF-Slim also provides two meta-operations called ```repeat```and ```stack```that allow users to repeatedly perform the same operation.

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

TF-Slim adds a new scoping mechanism called ```arg_scope``` used to simplify the code. 

````python 
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

`````

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

> Working Example : Specifying the VGG16 Layers 
png: ![VGG16](http://book.paddlepaddle.org/03.image_classification/image/vgg16.png)


```python
def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
  return net
```

## Training Models 

Training tensorflow models requires a model, a loss function, the gradient computation and a training routine that iteratively computes the gradients of the model weightsrelative to the loss ans updates the weights accordingly. 

### Losses 
The loss function define a quatity that we want to minimize. 
* For classification problems : cross entropy between the true distribution and the predicted probabilty distribution across classes 
* For regression problems : sum-of-squares differences between the predicted and the true value. 
* For multi-task learning models : use of multiple loss functions simultaneoulsly (sum of various loss functions) 

TF-Slim provides an easy to use mechanism for defining and keeping track of the loss funtions via the losses module: ```slim.lossses.softmax_cross_entropy```, ```slim.losses.sum_of_square```,```slim.losses.get_total_loss```.

> Standard classification loss 
```python


#Import the data : 
import tensorflow as tf 
vgg=tf.contrib.slim.nets.vgg

# Load the images and labels 
images, labels=.....

#Create the model 
predictions, _= vgg.vgg_16(images)

#define the loss functions ans get the total loss 
loss=slim.losses.softmax_cross_entropy(predictions, labels) 

```

> Multi-tasks model that produces multiple outputs 

```python 

#Load the image and labels 
images, scene_labels, depth_labels=.....

#Create the model 
scene_predictions, depth_predictions=CreateMultiTaskModel(images)

#Define the loss functions ans get the total loss 
classification_loss= slim.losses.softmax_cross_entropy(scene_predictions,scene_labels)
sum_of_square_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)

#The following two lines have the same effect: 
total_loss=classification_loss + sum_of_square_loss
total_loss=slim.losses.get_total_loss(add_regularization_losses=False)

```

### Training Loop 
TF-Slim provides a Train function that repeatedly measures the loss, computes gradients and saves the model to disk. 
For example, once we've specified the model, the loss function and the optimization scheme, we can call ```slim.learning.create_train_op``` and ```slim.learning.train``` to perform the optimization. 

> Working Example : Training the VGG16 Layers

```python
with tf.Graph().as_default():
#
images, labels =....


    
      
      
      
      
      






