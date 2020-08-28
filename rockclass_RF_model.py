# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:48:47 2020

@author: Vikram
"""


import pandas as pd
import numpy as np
import pickle

        

df = pd.read_csv("rockclass.csv", usecols=['D', 'H', 'Q', 'k', 'Class'])
num_columns = ['D','H','Q','k']
X = df.drop(columns=['Class'], axis=1)
Y = df.Class
#print(X.shape, Y.shape)
#print(X.head())

#importing the module 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df['Class_encoded'] = df['Class'].map({'minor':0, 'mild':1, 'sever':2})
X = df[['D', 'H', 'Q', 'k']]
y = df['Class_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

#for graph purpose
y_test_arr= y_test.to_numpy()

#model selection using pipeline
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#based on bagging the data
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=1000,
max_samples=50, bootstrap=True, n_jobs=-1, oob_score=True)

bag_clf.fit(X_train, y_train)


#prediction funtion
def prediction(classifier):
    y_pred1= classifier.predict(X_test)
    return y_pred1
    
#Y_pred is in array format, so we are changing y_test into array to make it visible in plot
y_pred= prediction(bag_clf)

#visualization of the analysis
from sklearn.metrics import accuracy_score
print(accuracy_score( y_test_arr, y_pred, normalize=False))


def accuracy_score(classifier):
    print("Training score : {:2f}".format(classifier.score(X_train,y_train)))
    print("Test score : {:2f}".format(classifier.score(X_test,y_test_arr)))

accuracy_score(bag_clf)

import matplotlib.pyplot as plt
def figure(test, pred):
    plt.title("Test vs Pred plot")
    plt.plot(pred, marker = 'o', color='g', label='pred')
    plt.plot(test, marker = '>', color='r', label='test')
    plt.legend()
    plt.xlabel("Observation")
    plt.ylabel("Rock Class")
    plt.grid()


#model analysis
graph1 = figure(y_test_arr, y_pred) 


import pickle
pickle_out = open("Rockclass.pkl", 'wb')
pickle.dump(bag_clf, pickle_out)
pickle_out.close()






