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
from sklearn.tree import DecisionTreeClassifier

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print(len(features_train[0])) ## no of columns in the features_train data [0]-> first row
clf=DecisionTreeClassifier(min_samples_split=40)
t0=time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0=time()
predictions = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)



#########################################################
### your code goes here ###


#########################################################


