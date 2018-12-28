# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:02:49 2018

@author: akash.sharma
"""
import numpy as np

class Perceptron:
    def __init__(self, N, alpha =0.01):
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha
        
        
    def step(fit, x):
        return 1 if x>0 else 0
    
    
    def fit(self, X, y, epochs = 10):
        X = np.c_[X, np.ones((X.shape[0]))]
        
        for epoch in np.arange(0,epochs):
            
            for (x,target) in zip(X,y):
                
                p = self.step(np.dot(x, self.W))
                
                if p!= target:
                    error = p - target
                    self.W += -self.alpha * error * x
                    
    def predict(self, X, addBias = True):
        
        X = np.atleast_2d(X)
        
        if addBias:
            
            X = np.c_[X, np.ones((X.shape[0]))]
            
            return self.step(np.dot(X, self.W))
        
if __name__ == '__main__':
    
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [1]])
    
    print("[INFO] trianing perceptron...")
    
    p = Perceptron(X.shape[1], alpha = 0.01)
    p.fit(X, y, epochs = 25)
    
    print("[INFO] teesting perceptron....")
    
    
    for (x, target) in zip(X,y):
        pred = p.predict(x)
        print("[INFO] data = {}, ground-truth = {}, pred = {}".format(x, target[0], pred))