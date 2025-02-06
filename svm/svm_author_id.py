#!/usr/bin/python3

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
from sklearn import svm
import numpy as np

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

# C_values = [10.0, 100.0, 1000.0, 10000.0]

# for C in C_values:
# print(f"\nTesting SVM with RBF kernel and C = {C}")
svc = svm.SVC(kernel="rbf", C=1000.0)
t0=time()
svc.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0=time()
predictions = svc.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# answer_10 = predictions[10]
# answer_26 = predictions[26]
# answer_50 = predictions[50]

# print(f"Prediction for element 10: {answer_10}")
# print(f"Prediction for element 26: {answer_26}")
# print(f"Prediction for element 50: {answer_50}")

num_chris = np.sum(predictions == 1)
print(f"Number of test predictions labeled as 1, predicted to be Chris: {num_chris}" )

accuracy = svc.score(features_test, labels_test)
print("Accuracy", accuracy)

#########################################################
