#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("Support Vector Machine")

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

classifier = svm.SVC(C=10000., kernel="rbf",gamma="scale")
t1 = time()
classifier.fit(features_train,labels_train)
print("Training time: ", round(time()-t1,3), " seconds")

"""
Linear kernel: 
Training time:  0.096  seconds
Prediction time:  0.947  seconds
Accuracy on 1% of set:  0.8845278725824801
"""

"""
Rbf kernel: 
Training time:  0.102  seconds
Prediction time:  0.958  seconds
Accuracy on 1% of set:  0.8953356086461889
Parametar C: 10 - > 0.8998862343572241
Parametar C: 100 - > 0.8998862343572241
Parametar C: 1000 - > 0.8998862343572241
Parametar C: 10000 -> 0.8998862343572241
"""

t=time()
predicted = classifier.predict(features_test)
print("Prediction time: ", round(time()-t,3), " seconds")

"""
Prediction for # elements in dataset
10th:
26th:
50th:
"""
pred10 = predicted[10]
pred26 = predicted[26]
pred50 = predicted[50]

print(pred10)
print(pred26)
print(pred50)

count = 0;
for i in predicted:
    if i == 1:
        count+=1
print(count)

# parsed_predicted_values = np.reshape(np.array(predicted), (-1, 2))
# parsed_real_values = np.reshape(np.array(labels_test), (-1, 2))
# print(classifier.score(parsed_predicted_values, parsed_real_values))

print("Accuracy: ", accuracy_score(labels_test,predicted))


#########################################################
### your code goes here ###

#########################################################


