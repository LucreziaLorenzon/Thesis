#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import SVC
import csv
from sklearn.utils import shuffle 
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
import itertools
import scipy.stats 
from time import time
import os
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#%%

#IMPORT DATASET--TRAINING--
SEL = [0, 0.0075]
NITER = 10000
NREF = 10000
#test_size --> percentage of samples destined to validation
test_size = 0.1
saving_path = "/home/lucrezialorenzon/RESULT_3/SVM/SVM_"+str(SEL[1])+"/"
#Dataset made of X and y (labels)
#X[0]-->Tajima's D X[1]-->nsl 
#y --> 0/1, are the two classes
num_samples = NITER*2
X = np.zeros((num_samples,2))
y = np.zeros((num_samples))
row_index = 0
for i,S in enumerate(SEL):
    with open("/home/lucrezialorenzon/RESULT_3/SVM/summarystatistics.csv") as csvfile:
         readCSV = csv.reader(csvfile)
         start = 0
         for row in readCSV:
             if start == 0:
                start = 1
             elif row[0] == str(S) and row[1] == str(NREF):
                  X[row_index,0] = row[3]
                  X[row_index,1] = row[5]
                  row_index += 1
#LABELING
y[10000:] = 1
#shuffle            
train_data = shuffle(X,y,random_state=2)
#Whole dataset
X = train_data[0]
y = train_data[1]
del train_data
#Splitting of X and y into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
del X,y
#PRE-PROCESSING OF THE DATASET
#standardization (the features becomes standard normally distributed,Gaussian with zero mean and variance = 1)
#we scale first on the train dataset and than we apply the same transformation on the test dataset
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)
np.save(saving_path +"X_test",X_test,allow_pickle=False)
np.save(saving_path +"y_test",y_test,allow_pickle=False)

#SELECTION OF THE BEST CLASSIFIER: RANDOMIZED SEARCH
param_grid = {'gamma': scipy.stats.expon(scale=.1),
                    'C': scipy.stats.expon(scale=100)}
clf_random_exp = RandomizedSearchCV(SVC(kernel='rbf'), param_distributions=param_grid,n_iter=100)
start = time()
clf_random_exp.fit(X_train, y_train)
print()
print("RandomizedSearchCV (exponencial distribution) took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), 100))
print("The best parameters set found with RandomizedSearchCV (exponencial distribution) are %s with a score of %0.2f"
      % (clf_random_exp.best_params_, clf_random_exp.best_score_))
#SVM linear 
param_C =  clf_random_exp.best_params_["C"]
param_C = round(param_C, 3)
param_gamma =  clf_random_exp.best_params_["gamma"]
param_gamma = round(param_gamma, 3)

#TRAINING 
clf = SVC(kernel='rbf', C=param_C, gamma=param_gamma)
clf.fit(X_train,y_train)

#SAVE MODEL
joblib.dump(clf, saving_path+"SVM_model")

#DECISION SURFACE PLOT
h = .02  # step size in the mesh
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.title('Support Vector Machine with polynomial kernel')
plt.xlabel("Tajima's D")
plt.ylabel("nSL")
plt.axis('tight')
blue_patch = mpatches.Patch(color=plt.cm.Paired(0.0), label='S = 0')
brown_patch = mpatches.Patch(color=plt.cm.Paired(1.0), label='S = '+str(S))
plt.legend(handles=[brown_patch,blue_patch])
plt.savefig(saving_path+"Decision_surface.eps")

#EVALUATION
y_pred=clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
loss = log_loss(y_test,y_pred)
print("Accuracy: " + str(accuracy))
print("Loss: " + str(loss))

#CLASSIFICATION REPORT
classes = ['S = 0','S = '+str(S)]
cr = classification_report(y_test,y_pred,target_names=classes)
print()
print("Calassification report")
print(cr)
np.save(saving_path +"Classification_report",cr,allow_pickle=False)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
fig = plt.figure(facecolor='white')
title='Normalized confusion matrix'
cmap=plt.cm.Blues
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f' 
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), 
             horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(saving_path+"confusion_matrix.eps",bbox_inches='tight')

del X_test,X_train,y_test,y_train,Z,cm,cr,fmt,h,i,j,loss,param_grid,row,row_index,start,thresh,title,x_max,x_min,xx,y_max,y_min,y_pred,yy,tick_marks
#%%

#TESTING
SEL = [0, 0.0075]
NITER = 10000
#NREF --> population size that we test
NREF = 50000
#test_size --> percentage of samples destined to validation
num_samples = 2*NITER
test_size = 0.1

os.mkdir("/home/lucrezialorenzon/RESULT_3/SVM/SVM_"+str(SEL[1])+"/Test_"+str(NREF))
saving_path = "/home/lucrezialorenzon/RESULT_3/SVM/SVM_"+str(SEL[1])+"/Test_"+str(NREF)+"/"
test_samples = int(NITER*2*test_size)

# LOAD MODEL
model_path = "/home/lucrezialorenzon/RESULT_3/SVM/SVM_"+str(SEL[1])+"/"
clf = joblib.load(model_path+"SVM_model")
#Find test samples
X = np.zeros((num_samples,2))
row_index = 0
for i,S in enumerate(SEL):
    with open("/home/lucrezialorenzon/RESULT_3/SVM/summarystatistics.csv") as csvfile:
         readCSV = csv.reader(csvfile)
         start = 0
         for row in readCSV:
             if start == 0:
                start = 1
             elif row[0] == str(S) and row[1] == str(NREF):
                  X[row_index,0] = row[3]
                  X[row_index,1] = row[5]
                  row_index += 1
#LABELING
y_test = np.zeros((test_samples))
y_test[1000:] = 1
X_test = np.zeros((test_samples,2))
X_test[:1000,:]=X[:1000,:]
X_test[1000:,:]=X[10000:11000,:]
#scaling
print(scaler)
X_test = scaler.transform(X_test)
#shuffle            
train_data = shuffle(X_test,y_test,random_state=2)
#Whole dataset
X_test = train_data[0]
y_test = train_data[1]
np.save(saving_path +"X_test",X_test,allow_pickle=False)
np.save(saving_path +"y_test",y_test,allow_pickle=False)

#PREDICTION
y_pred=clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
loss = log_loss(y_test,y_pred)
print("Accuracy: " + str(accuracy))
print("Loss: " + str(loss))

#CLASSIFICATION REPORT
classes = ['S = 0','S = '+str(S)]
cr = classification_report(y_test,y_pred,target_names=classes)
print()
print("Classification report")
print(cr)
np.save(saving_path +"Classification_report",cr,allow_pickle=False)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
fig = plt.figure(facecolor='white')
title='Normalized confusion matrix'
cmap=plt.cm.Blues
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f' 
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), 
             horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(saving_path+"confusion_matrix.eps",bbox_inches='tight')

#%%