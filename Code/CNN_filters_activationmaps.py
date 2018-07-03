#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:07:48 2018

@author: lucrezialorenzon
"""
#%%
import numpy as np
import theano
import keras
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
#%%

#LOAD MODEL
N = 1
path = "/home/lucrezialorenzon/RESULT_5/CNN_41/Training_"+str(N)+"/"
CNN = load_model(path+"CNN_model.h5")
X_train = np.load(path+"X_train.npy")
X_test = np.load(path+"X_test.npy")
y_test = np.load(path+"y_test.npy")

CNN.summary()

#%%

#VISUALIZING FILTERS OF THE FIRST CONVOLUTION

#Plot all kernels
#choose which convolution layer in CNN.layers[]
layer = CNN.layers[0]
kernels = layer.kernel.get_value()
kernels = kernels.reshape(32,3,3)
fig = plt.figure()
for j in range(len(kernels)):
    ax = fig.add_subplot(4,8,j+1)
    ax.matshow(kernels[j], cmap = matplotlib.cm.binary)
    plt.title(str(j+1),fontsize=8)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt.savefig(path+"Kernels.eps",bbox_inches='tight')

#VISUALIZING INTERMEDIATE LAYERS

output_layer = layer.output
#Don't change the following
output_fn = theano.function([layer.input],CNN.layers[2].output)
#Choose input image 
input_image = X_train[1:2,:,:,:]
fig = plt.figure()
plt.imshow(input_image[0,0,:,:],cmap='gray')
plt.savefig(path+"training_image.eps",bbox_inches='tight')

output_image = output_fn(input_image)
#Rearrange the dimensions so we can plot the result as an RGB image
output_image = np.rollaxis(np.rollaxis(output_image,3,1),3,1)

#Run to choose which filter we want to take 
#(remember that i is the filter number-1)
fig = plt.figure()
filters = 32
for i in range (filters):
    ax = fig.add_subplot(4,8,i+1)
    #ax.imshow(output_image[0,:,:,i],interpolation = 'nearest') #to see the firt filter
    ax.imshow(output_image[0,:,:,i],cmap=plt.cm.rainbow)
    plt.title(str(i+1),fontsize=8)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt.savefig(path+"Image_through_filters.eps",bbox_inches='tight')
plt.figure()
#choose which filters we want to see (select i)
i = 3
fig = plt.figure()
plt.imshow(output_image[0,:,:,(i-1)],cmap=plt.cm.rainbow)
plt.savefig(path+"Image_through_filter_"+str(i)+".eps",bbox_inches='tight')

#%%

channels = 1
nb_classes = 41
batch_size = 32
epochs = 20

kernel_size = 3
pooling_size = 2

NREF = 10000 
NITER = 1000
NCHROMS = 100
#image size --> (channels, width, height)
channels = 1
#should be changed each time...it takes the mean of dim (from the MS_training script)
img_columns = 122
img_rows = NCHROMS

#%%
