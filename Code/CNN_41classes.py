#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:58:29 2018

@author: lucrezialorenzon
"""

#%%

import numpy as np
import theano
import keras
import os
import shutil

from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils #Utilities that help in the data transformation
from keras import backend as K 
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first') #e' quello giusto per Theano
print("Image_data_format is " + K.image_data_format())
from keras.models import load_model

from PIL import Image

from sklearn.utils import shuffle 
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import rcParams
import itertools

#%%

#DATASET ELABORATION

#41 classes from 0 to 0.01
selection_coefficients = [0 ,  0.00025,  0.0005 ,  0.00075,  0.001  ,  0.00125,
        0.0015 ,  0.00175,  0.002  ,  0.00225,  0.0025 ,  0.00275,
        0.003  ,  0.00325,  0.0035 ,  0.00375,  0.004  ,  0.00425,
        0.0045 ,  0.00475,  0.005  ,  0.00525,  0.0055 ,  0.00575,
        0.006  ,  0.00625,  0.0065 ,  0.00675,  0.007  ,  0.00725,
        0.0075 ,  0.00775,  0.008  ,  0.00825,  0.0085 ,  0.00875,
        0.009  ,  0.00925,  0.0095 ,  0.00975,  0.01]

NREF = 10000 
NITER = 1000
NCHROMS = 198
#image size --> (channels, width, height)
channels = 1
#should be changed each time...it takes the mean of dim (from the MS_training script)
img_columns = 151
img_rows = NCHROMS

#%%
#RESIZE

path = "/home/lucrezialorenzon/RESULT_5/Images/"
listing = os.listdir(path)
num_samples = len(listing)
#Open image and resize --> (img_rows,img_columns)
for file in listing:
    im = Image.open(path + file)
    #N.B. im.resize wants as arguments (width,heigth)
    im_resized = im.resize((img_columns,img_rows))
    im_resized.save(path + file)
    #possibility jpeg convercion: im_resized.save(path1 + '/' + file, "JPEG")
    
#ARCHITECTURE

#Parameters
channels = 1
nb_classes = 41
#batch_size --> number of samples used per gradient update (default = 32)
#epochs --> number of epochs to train the model; an epoch is an iteration over the
#entire x and y data provided
batch_size = 32
epochs = 20
filters = 32
kernel_size = 3
pooling_size = 2

CNN = Sequential()
#32 convolution filters, 3x3 Kernel.
#The step size is (1,1) by default, can be changed with the 'subsample' parameter.
#padding="same" results in padding the input such that the output has the same length as the original input. 
CNN.add(Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_first', input_shape=(channels,img_rows,img_columns)))
CNN.add(MaxPooling2D(pool_size=(pooling_size,pooling_size)))
CNN.add(Convolution2D(filters,(kernel_size, kernel_size), strides=(1,1), activation='relu',padding='same', data_format='channels_first'))
CNN.add(MaxPooling2D(pool_size=(pooling_size,pooling_size)))
#Dropout prevents overfitting
#rate --> float between 0 and 1; it is the fraction of the input units to drop.
CNN.add(Dropout(rate=0.5)) 
#The weights from the Convolution layers must be flattened (made 1-dimensional) 
#before passing them to the fully connected Dense layer
CNN.add(Flatten())
#Densely-connected NN layer
CNN.add(Dense(128, activation='relu'))
CNN.add(Dropout(rate=0.5)) 
CNN.add(Dense(nb_classes, activation='softmax'))

#Prints a summary representation of the CNN
CNN.summary()
#Compilation of the model: configures the CNN to get ready for training
CNN.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

#FIRST TRAINING 
NITER = 1000
n=1
os.mkdir("/home/lucrezialorenzon/RESULT_5/CNN_41/Training_"+str(n))
    
#MOVING IMAGES
#training images moving--we take the first 3000 images for each class
#path --> where training images will be moved
path = "/home/lucrezialorenzon/RESULT_5/Images_training/"
os.mkdir("/home/lucrezialorenzon/RESULT_5/Images_training")

start = (n - 1)*NITER
for S in selection_coefficients:
    source = "/home/lucrezialorenzon/RESULT_5/Images/"
    destination = path
    for ITER in range(NITER):
        file = "img.." + str(S) + ".."+ str(start+ITER+1) + ".bmp"
        shutil.copy(source + file,destination)
 
saving_path = "/home/lucrezialorenzon/RESULT_5/CNN_41/Training_"+str(n)+"/"

#ELABORATE DATASET    
#Creation of a list in which element is the name of a file contained in Dataset folder
#num_samples --> number of images in the dataset
listing = os.listdir(path)
num_samples = len(listing)

im_matrix_rows = len(selection_coefficients)*NITER
im_matrix_columns = img_rows*img_columns
im_matrix = np.empty((im_matrix_rows,im_matrix_columns), dtype='float32')
index = 0
#takes the images in the order given by the intestation
for s in selection_coefficients:
    intestation = "img.."+str(s)+".."
    #Flattening of each image and creation of im_matrix, the array that contains all image
    for i,im in enumerate(listing):
        if intestation in listing[i]:
           image = np.asarray(Image.open(path + im)).flatten()
           image = image.astype('float32')
           im_matrix[index,:] = image
           index += 1
#Labeling
#we have x-1 samples under S = 1 [N.B. 0:x --> the xth is not taken]
label = np.zeros((num_samples,),dtype=int)
start = 0
stop = NITER
for i in range(len(selection_coefficients)):
    label[start:stop] = i
    start = start + NITER
    stop = stop + NITER
#Shuffle
im_matrix,label = shuffle(im_matrix,label,random_state=2)

#Splitting of X and y into training and testing
#test_size --> percentage of samples destined to validation
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(im_matrix, label, test_size=test_size, random_state=4)

#X_train.shape[0] --> number of samples used for training
X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_columns)
X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_columns)
#Float convercion and normalization
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#If it doesn't work (ErrorMemory) try with for cycles
#for i in range(160000):
    #X_train[i,:,:,:] = X_train[i,:,:,:].astype('float32')
#for i in range(160000):
    #X_train[i,:,:,:] = X_train[i,:,:,:]/255
    
#convert class vectors to binary class matrices
nb_classes = 41
y_train= np_utils.to_categorical(y_train,nb_classes)
y_test= np_utils.to_categorical(y_test,nb_classes)

np.save(saving_path+"X_train",X_train,allow_pickle=False)
np.save(saving_path+"X_test",X_test,allow_pickle=False)
np.save(saving_path+"y_train",y_train,allow_pickle=False)
np.save(saving_path+"y_test",y_test,allow_pickle=False)
del im_matrix, label, S, file, im, i, intestation, listing, s, start, stop, image

#TRAINING
#verbose --> Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
hist = CNN.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
               verbose=1, validation_data=(X_test, y_test))

#SAVE WHOLE MODEL (architecture + weights + training configuration[loss,optimizer] +
# state of the optimizer). This allows to resume training where we left off.
CNN.save(saving_path +"CNN_model.h5")

#LOSSES AND ACCURACIES --> PLOTS
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(epochs)
x_axis =np.zeros(len(xc))
for x,i in enumerate (xc):
    x_axis[i]=x+1

rcParams['axes.titlepad'] = 20 
plt.figure(1,figsize=(7,5),facecolor='white')
plt.plot(x_axis,train_loss)
plt.plot(x_axis,val_loss)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training loss and validation loss',fontsize=12)
plt.grid(True)
plt.legend(['Training loss','Validation loss'],fontsize=12)
plt.style.use(['classic'])
plt.savefig(saving_path + "Loss.eps")
plt.close()

plt.figure(2,figsize=(7,5),facecolor='white')
plt.plot(x_axis,train_acc)
plt.plot(x_axis,val_acc)
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Accuracy',fontsize=12) 
plt.title('Training accuracy and validation accuracy',fontsize=12)
plt.grid(True)
plt.legend(['Training accuracy','Validation accuracy'],fontsize=12,loc=4)
plt.style.use(['classic'])
plt.savefig(saving_path + "Accuracy.eps")
plt.close()

#CONFUSION MATRIX
# Compute confusion matrix
Y_pred = CNN.predict(X_test,batch_size=None, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

classes = np.zeros(41)
for i in range(41):
    classes[i] =  str(selection_coefficients[i])
classes = classes.astype('str')

cm = confusion_matrix(np.argmax(y_test,axis=1), y_pred)
np.set_printoptions(precision=2)
fig = plt.figure(facecolor='white')
title='Normalized confusion matrix'
cmap=plt.cm.Blues
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
#plt.colorbar(shrink=0.7)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
#plt.xticks(tick_marks, rotation=45, fontsize=6)
plt.yticks(tick_marks, classes, fontsize=8)
#plt.yticks(tick_marks)
#fmt = '.2f' 
#thresh = cm.max() / 2.
#for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    plt.text(j, i, format(cm[i, j], fmt), 
#             horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(saving_path+"confusion_matrix.eps",bbox_inches='tight')

#CLASSIFICATION REPORT
#Note that in binary classification, recall of the positive class is also known as “sensitivity”; 
#recall of the negative class is “specificity”.
cr = classification_report(np.argmax(y_test,axis=1),y_pred, target_names = classes)
print(cr)
np.save(saving_path +"classification_report",cr,allow_pickle=False)


np.save(saving_path +"val_acc",val_acc,allow_pickle=False)
np.save(saving_path +"val_loss",val_loss,allow_pickle=False)
np.save(saving_path +"train_acc",train_acc,allow_pickle=False)
np.save(saving_path +"train_loss",train_loss,allow_pickle=False)

#%%

#ADD TRAINING 

#Parameters
channels = 1
nb_classes = 41
#batch_size --> number of samples used per gradient update (default = 32)
#epochs --> number of epochs to train the model; an epoch is an iteration over the
#entire x and y data provided
batch_size = 32
epochs = 20
filters = 32
kernel_size = 3
pooling_size = 2

for n in [3,4,5,6,7,8,9,10]:
    os.mkdir("/home/lucrezialorenzon/RESULT_5/CNN_41/Training_"+str(n))
    
    #LOAD MODEL
    CNN = load_model("/home/lucrezialorenzon/RESULT_5/CNN_41/Training_"+str(n-1)+"/CNN_model.h5")
    
    #MOVING IMAGES
    #training images moving--we take the first 3000 images for each class
    #path --> where training images will be moved
    path = "/home/lucrezialorenzon/RESULT_5/Images_training/"
    shutil.rmtree("/home/lucrezialorenzon/RESULT_5/Images_training")
    os.mkdir("/home/lucrezialorenzon/RESULT_5/Images_training")
    
    start = (n - 1)*NITER
    for S in selection_coefficients:
        source = "/home/lucrezialorenzon/RESULT_5/Images/"
        destination = path
        for ITER in range(NITER):
            file = "img.." + str(S) + ".."+ str(start+ITER+1) + ".bmp"
            shutil.copy(source + file,destination)
     
    saving_path = "/home/lucrezialorenzon/RESULT_5/CNN_41/Training_"+str(n)+"/"
    
    #ELABORATE DATASET    
    #Creation of a list in which element is the name of a file contained in Dataset folder
    #num_samples --> number of images in the dataset
    listing = os.listdir(path)
    num_samples = len(listing)
    
    im_matrix_rows = len(selection_coefficients)*NITER
    im_matrix_columns = img_rows*img_columns
    im_matrix = np.empty((im_matrix_rows,im_matrix_columns), dtype='float32')
    index = 0
    #takes the images in the order given by the intestation
    for s in selection_coefficients:
        intestation = "img.."+str(s)+".."
        #Flattening of each image and creation of im_matrix, the array that contains all image
        for i,im in enumerate(listing):
            if intestation in listing[i]:
               image = np.asarray(Image.open(path + im)).flatten()
               image = image.astype('float32')
               im_matrix[index,:] = image
               index += 1
    #Labeling
    #we have x-1 samples under S = 1 [N.B. 0:x --> the xth is not taken]
    label = np.zeros((num_samples,),dtype=int)
    start = 0
    stop = NITER
    for i in range(len(selection_coefficients)):
        label[start:stop] = i
        start = start + NITER
        stop = stop + NITER
    #Shuffle
    im_matrix,label = shuffle(im_matrix,label,random_state=2)
    
    #Splitting of X and y into training and testing
    #test_size --> percentage of samples destined to validation
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(im_matrix, label, test_size=test_size, random_state=4)
    
    #X_train.shape[0] --> number of samples used for training
    X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_columns)
    X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_columns)
    #Float convercion and normalization
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    #If it doesn't work (ErrorMemory) try with for cycles
    #for i in range(160000):
        #X_train[i,:,:,:] = X_train[i,:,:,:].astype('float32')
    #for i in range(160000):
        #X_train[i,:,:,:] = X_train[i,:,:,:]/255
        
    #convert class vectors to binary class matrices
    nb_classes = 41
    y_train= np_utils.to_categorical(y_train,nb_classes)
    y_test= np_utils.to_categorical(y_test,nb_classes)
    
    np.save(saving_path+"X_train",X_train,allow_pickle=False)
    np.save(saving_path+"X_test",X_test,allow_pickle=False)
    np.save(saving_path+"y_train",y_train,allow_pickle=False)
    np.save(saving_path+"y_test",y_test,allow_pickle=False)
    del im_matrix, label, S, file, im, i, intestation, listing, s, start, stop, image
    
    #TRAINING
    #verbose --> Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
    hist = CNN.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
                   verbose=1, validation_data=(X_test, y_test))
    
    #SAVE WHOLE MODEL (architecture + weights + training configuration[loss,optimizer] +
    # state of the optimizer). This allows to resume training where we left off.
    CNN.save(saving_path +"CNN_model.h5")
    
    #LOSSES AND ACCURACIES --> PLOTS
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(epochs)
    x_axis =np.zeros(len(xc))
    for x,i in enumerate (xc):
        x_axis[i]=x+1
    
    rcParams['axes.titlepad'] = 20 
    plt.figure(1,figsize=(7,5),facecolor='white')
    plt.plot(x_axis,train_loss)
    plt.plot(x_axis,val_loss)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training loss and validation loss',fontsize=12)
    plt.grid(True)
    plt.legend(['Training loss','Validation loss'],fontsize=12)
    plt.style.use(['classic'])
    plt.savefig(saving_path + "Loss.eps")
    plt.close()
    
    plt.figure(2,figsize=(7,5),facecolor='white')
    plt.plot(x_axis,train_acc)
    plt.plot(x_axis,val_acc)
    plt.xlabel('Epoch',fontsize=12)
    plt.ylabel('Accuracy',fontsize=12) 
    plt.title('Training accuracy and validation accuracy',fontsize=12)
    plt.grid(True)
    plt.legend(['Training accuracy','Validation accuracy'],fontsize=12,loc=4)
    plt.style.use(['classic'])
    plt.savefig(saving_path + "Accuracy.eps")
    plt.close()
    
    #CONFUSION MATRIX
    # Compute confusion matrix
    Y_pred = CNN.predict(X_test,batch_size=None, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    classes = np.zeros(41)
    for i in range(41):
        classes[i] =  str(selection_coefficients[i])
    classes = classes.astype('str')
    
    cm = confusion_matrix(np.argmax(y_test,axis=1), y_pred)
    np.set_printoptions(precision=2)
    fig = plt.figure(facecolor='white')
    title='Normalized confusion matrix'
    cmap=plt.cm.Blues
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #plt.colorbar(shrink=0.7)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    #plt.xticks(tick_marks, rotation=45, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=8)
    #plt.yticks(tick_marks)
    #fmt = '.2f' 
    #thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt), 
    #             horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(saving_path+"confusion_matrix.eps",bbox_inches='tight')
    
    #CLASSIFICATION REPORT
    #Note that in binary classification, recall of the positive class is also known as “sensitivity”; 
    #recall of the negative class is “specificity”.
    cr = classification_report(np.argmax(y_test,axis=1),y_pred, target_names = classes)
    print(cr)
    np.save(saving_path +"classification_report",cr,allow_pickle=False)
    
    
    np.save(saving_path +"val_acc",val_acc,allow_pickle=False)
    np.save(saving_path +"val_loss",val_loss,allow_pickle=False)
    np.save(saving_path +"train_acc",train_acc,allow_pickle=False)
    np.save(saving_path +"train_loss",train_loss,allow_pickle=False)
    del X_train,X_test,y_train,y_test
    del cr,val_acc,val_loss,train_acc,train_loss,tick_marks,cm,classes,Y_pred,y_pred,xc,x_axis,hist
#%%

