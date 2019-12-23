#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import matplotlib.pyplot as plt
import sys
from time import time
import matplotlib.pyplot as plt
from matplotlib import style
sys.path.append("../tools/")
from email_preprocess import preprocess

style.use("ggplot")

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("Naive Bayes")
#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

classifier = GaussianNB()

#split a set to 1% to increase speed
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
classifier.fit(features_train, labels_train)
print("Training time:", round(time()-t0,3),"s")

t1 = time()
predicted = classifier.predict(features_test)
print("Prediction time:", round(time()-t1,3),"s")

# accuracy = number of points classified correctly / all points
# print(classifier.score(labels_test, predicted, sample_weight=None))
print(accuracy_score(labels_test,predicted))
#########################################################
