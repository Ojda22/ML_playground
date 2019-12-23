#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import pandas as pd
import numpy as np



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# df_mails = pd.DataFrame(data=features_train)
# print(df_mails)

#########################################################
### your code goes here ###

from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# param -> min_sample_split parametar decide when to stop splitting--branching a tree
# param -> entropy parametar controls where DT decides where to split the data
#          def: measure of impurity in a bunch of examples

# classifier1 = tree.DecisionTreeClassifier(min_samples_split=2)
classifier2 = tree.DecisionTreeClassifier(min_samples_split=40)
# classifier1 = classifier1.fit(features_train, labels_train)
classifier2 = classifier2.fit(features_train, labels_train)
# predicted1 = classifier1.predict(features_test)
predicted2 = classifier2.predict(features_test)

# acc_for_split_2 = accuracy_score(predicted1, labels_test)
acc_for_split_50 = accuracy_score(predicted2, labels_test)
### be sure to compute the accuracy on the test set
# print(round(acc_for_split_2,8))
print(round(acc_for_split_50,8))
print(len(features_train[0]))


#########################################################


