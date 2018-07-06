# Convolutional Neural Networks for the identification of signatures of natural selection and functional mutations in the human genome

## realdata_FASTA.py

###### Script to convert real genetic allignment data into images; it produces three multi-population images and three single-population images for each population contained in the file .FASTA. 

REQUIREMENTS: Python 3.6

INPUT: Two FASTA files that contain allignment genetic sequences; a reference genome (gene\_anc.fasta) and several genomes of the populations (gene\_gene.fasta). 

## MSMS_images.py

###### Script to convert simulated genetic allignment data into images; it produces a black and white image (derived/ancestral allele) for each simulation replicate.

REQUIREMENTS: Python 3.6

INPUT: Genetic simulations from MSMS software (http://www.mabs.at/ewing/msms/index.shtml)

## SVM.py

###### Script that implements a Support Vector Machine. 

INPUT: csv file that contains the summary statistics of interest, related to each MSMS simulation replicate.

It performs:

1. Dataset elaboration
2. Parameter optimization (C and gamma) with RandomizedSearchCV
3. Training of the classifier
4. Decision surface visualization
5. Model evaluation on a validation set (classification report and confusion matrix)
6. Testing on a test set (classification report and confusion matrix)

## CNN_2classes.py

###### Script that implements a Convolutional Neural Network for binary classification (neutrality/positive selection).

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
5. CNN testing 




