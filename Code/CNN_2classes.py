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

from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils #Utilities that help in the data transformation
from keras import backend as K 
K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first') #It is the right one for Theano
print("Image_data_format is " + K.image_data_format())

from PIL import Image

from sklearn.utils import shuffle 
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams
import itertools

#%%

#DATASET ELABORATION

selection_coefficients = [0, 0.0075]
NREF = [10000] 
NITER = 10000
NCHROMS = 100

#image size --> (channels, width, height)
path = "/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/Dataset_training/"
channels = 1
img_rows = NCHROMS
#should be changed each time...it takes the mean of dim (from the MS_training script)
img_columns = 272

#Creation of a list in which element is the name of a file contained in Dataset folder
#num_samples --> number of images in the dataset
listing = os.listdir(path)
num_samples = len(listing)

#Open image and resize --> (img_rows,img_columns)
for file in listing:
    im = Image.open(path + file)
    #N.B. im.resize wants as arguments (width,heigth)
    im_resized = im.resize((img_columns,img_rows))
    im_resized.save(path + file)
    #possibility jpeg convercion: im_resized.save(path1 + '/' + file, "JPEG")

im_matrix_rows = len(NREF)*len(selection_coefficients)*NITER
im_matrix_columns = img_rows*img_columns
im_matrix = np.zeros((im_matrix_rows,im_matrix_columns))
index = 0

intestation = ["img..0..","img..0.0075.."]
#takes the images in the order given by the intestration
for s in intestation:
    #Flattening of each image and creation of im_matrix, the array that contains all image
    for i,im in enumerate(listing):
        if s in listing[i]:
           im_matrix[index,:] = np.asarray(Image.open("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/Dataset_training/" + im)).flatten()
           index += 1

#Labeling
#we have x-1 samples under S = 1 [N.B. 0:x --> the xth is not taken]
label = np.ones((num_samples,),dtype=int)
label[0:10000] = 0
label[10000:] = 1

#Shuffle
#train_data[0] --> (num_samples, img_rows x img_columns)
#train_data[1] --> (num_samples)
train_data = shuffle(im_matrix,label,random_state=2)
del im_matrix, label
#Splitting of X and y into training and testing
#test_size --> percentage of samples destined to validation
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size=test_size, random_state=4)
del train_data
#X_train.shape[0] --> number of samples used for training
X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_columns)
X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_columns)
#Float convercion and normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#If it doesn't work (ErrorMemory) try with for cycles
#for i in range(160000):
    #X_train[i,:,:,:] = X_train[i,:,:,:].astype('float32')
#for i in range(160000):
    #X_train[i,:,:,:] = X_train[i,:,:,:]/255
    
#convert class vectors to binary class matrices
nb_classes = 2
y_train= np_utils.to_categorical(y_train,nb_classes)
y_test= np_utils.to_categorical(y_test,nb_classes)

np.save("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/X_train",X_train,allow_pickle=False)
np.save("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/X_test",X_test,allow_pickle=False)
np.save("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/y_train",y_train,allow_pickle=False)
np.save("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/y_test",y_test,allow_pickle=False)

#%%

#ARCHITECTURE


#Parameters
channels = 1
nb_classes = 2
#batch_size --> number of samples used per gradient update (default = 32)
#epochs --> number of epochs to train the model; an epoch is an iteration over the
#entire x and y data provided
batch_size = 1
epochs = 2
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
CNN.add(Dense(2, activation='softmax'))

#Prints a summary representation of the CNN
CNN.summary()
#Compilation of the model: configures the CNN to get ready for training
CNN.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

#%%

#TRAINING

#verbose --> Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
hist = CNN.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
               verbose=1, validation_data=(X_test, y_test))

#%%

#EVALUATION

score = CNN.evaluate(X_test, y_test, batch_size=None, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%

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
plt.savefig("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/Loss.png")

plt.figure(2,figsize=(7,5),facecolor='white')
plt.plot(x_axis,train_acc)
plt.plot(x_axis,val_acc)
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Accuracy',fontsize=12) 
plt.title('Training accuracy and validation accuracy',fontsize=12)
plt.grid(True)
plt.legend(['Training accuracy','Validation accuracy'],fontsize=12,loc=4)
plt.style.use(['classic'])
plt.savefig("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/Accuracy.png")
#%%

#CONFUSION MATRIX

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    rcParams['axes.titlepad'] = 20 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=12)
    plt.colorbar(shrink = 0.7)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize=12,
                 horizontalalignment="center", color="white" 
                 if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=12)
    plt.xlabel('Predicted label',fontsize=12)

# Compute confusion matrix
Y_pred = CNN.predict(X_test,batch_size=None, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
#p shows which is the posterior distribution of esch testing sample
#p is equal to Y_pred
p=CNN.predict_proba(X_test)
classes = ['S = 0','S = 0.0075']
cnf_matrix = confusion_matrix(np.argmax(y_test,axis=1), y_pred)
np.set_printoptions(precision=2)
plt.figure(figsize=(4,4),facecolor='white')
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
#%%

#SAVE WHOLE MODEL (architecture + weights + training configuration[loss,optimizer] +
# state of the optimizer). This allows to resume training where we left off.
CNN.save("/home/lucrezialorenzon/RESULT_2/row_col_ordered/0and0.0075_10000/CNN_model.h5")

del CNN

#LOAD MODEL
from keras.models import load_model
CNN = load_model("/home/lucrezialorenzon/RESULT_2/row_ordered/0and0.001_10000/CNN_model.h5")

#%%


                  

