'''
Implementation of 'SqueezeNet' recommended in 'SqueezeNet: AlexNet-Level 
accuracy with 50x fewer parameters and <0.5MB Model Size' by: Iandola et al.

Review jupyter notebook squeezenet_architecture.ipynb included in this repository if possible. 

If desired, place this .py in your working directory and call the fire module and SqueezeNet architeture to your workspace with

from squeezenet_architecture import fire_module, SqueezeNet
'''

# Load libraries

import tensorflow as tf
import numpy as np

# shortcut to layers and models provided in tf
layers = tf.contrib.keras.layers
models = tf.contrib.keras.models

# Create a function for the 'fire module' referenced in the paper. The fire module consists of a squeeze convolution layer
# (1x1 filters), feeding into an expand layer that has a mix of 1x1 and 3x3 convolution filters (Iandola 2016)

'''
fire_mod

Defines the architecture of the fire module

Parameters
----------
x : input
fire_id: id of the fire module 
squeeze : output feature maps from the squeeze layer (default = 16)
expand : output feature maps from expand layers (default = 64)

'''


def fire_mod(x, fire_id, squeeze=16, expand=64):
    
    # initalize naming convention of components of the fire module
    squeeze1x1 = 'squeeze1x1'
    expand1x1 = 'expand1x1'
    expand3x3 = 'expand3x3'
    relu = 'relu.'
    fid = 'fire' + str(fire_id) + '/'
    
    # define the squeeze layer ~ (1,1) filter
    x = layers.Convolution2D(squeeze, (1,1), padding = 'valid', name= fid + squeeze1x1)(x)
    x = layers.Activation('relu', name= fid + relu + squeeze1x1)(x)
    
    # define the expand layer's (1,1) filters
    expand_1x1 = layers.Convolution2D(expand, (1,1), padding='valid', name= fid + expand1x1)(x)
    expand_1x1 = layers.Activation('relu', name= fid + relu + expand1x1)(expand_1x1)
    
    # define the expand layer's (3,3) filters
    expand_3x3 = layers.Convolution2D(expand, (3,3), padding='same', name= fid + expand3x3)(x)
    expand_3x3 = layers.Activation('relu', name= fid + relu + expand3x3)(expand_3x3)
    
    # Concatenate
    x = layers.concatenate([expand_1x1, expand_3x3], axis = 3, name = fid + 'concat')
    
    return x
    
    

# Use the fire_mod (fire module) to construct the SqueezeNet architecture

'''
SqueezeNet

Implementation of the SqueezeNet architecture ~ constructed to expected inputs from CIFAR-10
which is 32x32x3 with 10 output classes. The paper is optimized for 224x224x3 inputs, but that
is inappropriate for the desired implementation. Also, smaller images in CIFAR-10 than paper's 
inputs, and therefore the depth is considerably less (original SqueezeNet has 9 fire modules.)

Parameters
----------
input_shape : input array/image, default from CIFAR-10 (32,32,3)
classes : output classes, default 10. 

'''


def SqueezeNet(input_shape = (32,32,3), classes = 10):
        
    img_input = layers.Input(shape=input_shape)
    
    x = layers.Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv')(img_input)
    x = layers.Activation('relu', name='relu_conv1')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_mod(x, fire_id=2, squeeze=16, expand=64)
    x = fire_mod(x, fire_id=3, squeeze=16, expand=64)

    x = fire_mod(x, fire_id=4, squeeze=32, expand=128)
    x = fire_mod(x, fire_id=5, squeeze=32, expand=128)
    x = layers.Dropout(0.5, name='drop9')(x)

    x = layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = layers.Activation('relu', name='relu_conv10')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(img_input, out, name='squeezenet')

    return model
    
    
