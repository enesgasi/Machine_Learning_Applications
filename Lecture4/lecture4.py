#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:03:20 2024

@author: mahmutbagci



import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[2, 4.8], [2.9, 4.7], [2.5, 5], [3.2, 5.5], [6, 5], [7.6, 4],
[3.2, 0.9], [2.9, 1.9], [2.4, 3.5], [0.5, 3.4], [1, 4], [0.9, 5.9]])
#y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
Classifier_LR = linear_model.LogisticRegression(solver='liblinear', C=75)
#try solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
#Read model parameters from the link below
#https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
Classifier_LR.fit(X, y)
def Logistic_visualize(Classifier_LR, X, y):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.02
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))
    output =Classifier_LR.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    print(output)
    print(len(output))
    print(len(output[0]))
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1,cmap=plt.cm.Paired)
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1),1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1),1.0)))
    plt.show()

Logistic_visualize(Classifier_LR, X, y)

"""



import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print(label_names)
print(feature_names)
print(features)
print(labels[0])

print(feature_names[0])

print(features[0])


from sklearn.model_selection import train_test_split

train, test, train_labels, test_labels=train_test_split(features,labels,test_size = 0.40, random_state = 42)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)


preds = gnb.predict(test)
print(preds)

print(accuracy_score(test_labels,preds))

precision1 = precision_score(test_labels,preds)
print(precision1)
        
recall1 = recall_score(test_labels,preds)
print(recall1)
        
f1score = f1_score(test_labels,preds)
print(f1score)

cm=confusion_matrix(test_labels,preds)
print(cm)