# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:05:59 2018

@author: akash.sharma
"""

# importing necessary packages 

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return(1.0/(1+np.exp(-x)))
    
def predict(X,W):
    preds = sigmoid_activation(X.dot(W))
    
    preds[preds>0.5] = 1
    preds[preds<=0.5] = 0
    
    return(preds)
    
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type = float, default = 100)
ap.add_argument("-a", "--alpha", type = float, default = 0.01)
args = vars(ap.parse_args())

(X,y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 1.5, random_state = 1)
y = y.reshape((y.shape[0], 1))


X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size = 0.5, random_state = 42)

print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    preds = sigmoid_activation(trainX.dot(W))
    
    error = preds - trainY
    loss = np.sum(error**2)
    losses.append(loss)
    
    gradient = trainX.T.dot(error)
    
    W += -args["alpha"]*gradient
    
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss = {:.7f}".format(int(epoch+1), loss))
        
preds = predict(testX, W)
print(classification_report(testY, preds))