#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 23:18:01 2019

@author: qingn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# %%
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# assign colum names to the dataset
names  = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# read dataset to panas dataframe
dataset = pd.read_csv(url, names = names)
dataset.head()
# split our dataset into attributes and labels 

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# pridictions and training

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# evaluating the algorithm
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# What is K
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
# Plot error values against K values
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
# %%
def find_closest(alist, target):
    return min(alist, key=lambda x:abs(x-target))

X = [ 84.04467948,  52.42447842,  39.13555678,  21.99846595]
Y = [ 78.86529444,  52.42447842,  38.74910101,  21.99846595]

def list_matching(list1, list2):
    list1_copy = list1[:]
    pairs = []
    for i, e in enumerate(list2):
        elem = find_closest(list1_copy, e)
        pairs.append([i, list1.index(elem)])
        list1_copy.remove(elem)
    return pairs
# %%
def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
# %%

A= np.arange(0, 20.)
target = np.array([[-2, 100., 2., 2.4, 2.5, 2.6]])
print(A)
find_closest(A, target)
find_closest(time_co1, time_cpc_1)
