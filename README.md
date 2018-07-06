# Convolutional Neural Networks for the identification of signatures of natural selection and functional mutations in the human genome

## realdata_FASTA.py

###### Script to convert real genetic allignment data into images; it produces three multi-population images and three single-population images for each population contained in the file .FASTA. 

REQUIREMENTS: Python 3.6

INPUT: Two FASTA files that contain allignment genetic sequences; a reference genome (gene\_anc.fasta) and several genomes of the populations (gene\_gene.fasta)

## MSMS_simulations.py

###### Simulates genetic data.

REQUIREMENTS: MSMS software (http://www.mabs.at/ewing/msms/index.shtml)

## MSMS_images.py

###### Script to convert simulated genetic allignment data into images; it produces a black and white image (derived/ancestral allele) for each simulation replicate.

REQUIREMENTS: Python 3.6

INPUT: Genetic simulations 

## MSMS_summarystatistics.py

###### Script that calculates the summary statistics of each simulation replicate. 

## SVM.py

###### Script that implements a Support Vector Machine. 

INPUT: csv file that contains the summary statistics of interest, related to each MSMS simulation replicate

It performs:

1. Dataset elaboration
2. Parameter optimization (C and gamma) with RandomizedSearchCV
3. Training of the classifier
4. Decision surface visualization
5. Model evaluation on a validation set (classification report and confusion matrix)
6. Testing 

## CNN_2classes.py

###### Script that implements a Convolutional Neural Network for binary classification (neutrality/positive selection).

REQUIREMENTS: 

* Python 3.5
* Keras 2.2.0
* Theano 1.0.0

INPUT: Genetic images from simulations (created with MSMS_images.py)

It performs:

1. Dataset elaboration
2. CNN model definition
3. CNN training
4. CNN evaluation 
* PLOT: training/validation loss function
* PLOT: training/validation accuracy
* Classification report and confusion matrix
5. CNN testing

## CNN_41classes.py

###### Script that implements a Convolutional Neural Network with 41 output classes of the selection coefficient.

REQUIREMENTS: 

* Python 3.5
* Keras 2.2.0
* Theano 1.0.0

INPUT: Genetic images from simulations (created with MSMS_images.py).

It performs:

1. Dataset elaboration
2. CNN model definition
3. CNN training
4. CNN evaluation 
* PLOT: training/validation loss function
* PLOT: training/validation accuracy
* Classification report and confusion matrix
5. CNN testing 

## CNN_filters_activationmaps.py

###### Visualization and evaluation of CNN learning process

INPUT: 
* CNN trained model
* A genetic image

OUTPUT:
* Image of the filters of the first Convolutional layer
* Image of the activation maps of the first Convolutional layer



