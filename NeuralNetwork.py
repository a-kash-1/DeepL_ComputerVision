# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:20:35 2018

@author: akash.sharma
"""

# import numpy as np

class NeuralNetwork:
    # initialize the list of weights matrices, then store the 
    # network architecture and learning rate
    def __init__(self, layers, alpha = 0.01):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        # Now initialising the weight matrix    
        for i in np.arange(0, len(layers) - 2):
            # randomly initialise weights matrix connecting the
            # number of nodes in each respective layer together,
            # adding extra node for the bias indicated in next line by + 1
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            # normalize the weights by dividing from square root of layer's weights
            self.W.append(w/np.sqrt(layers[i]))
        
        # the last two layers are a special case where the input term need a bias
        # but the output term do not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))
        
    
    # The next function is called "magic method" named "__repr__"
    # this function is useful for debugging
    def __repr__(self):
        # construct and return a string that
        # represents the network architecture
        return("NeuralNetwork: {}".format("-".join(str(l) for l in self.layers)))
    
    def sigmoid(self, x):
        # returns the sigmoid of x
        return(1.0/(1+np.exp(-x)))
    
    # The activation function chosen above should be differenctiable
    # in order for backpropagation to work
    def sigmoid_deriv(self,x):
        # return the derivative of the sigmoid function of x
        return x*(1-x)
    
    def fit(self, X, y, epochs = 1000, displayUpdate = 100):
        # column of 1's is inserted as the last entry in the feature matrix
        # to treat bias as a trainable parameter
        X = np.c_[X, np.ones(X.shape[0])]
        
        # loop of epochs to train the model
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
                
            if epoch == 0 or (epoch + 1)% displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))

    def fit_partial(self, x, y):
        # construct list of activations for each layer
        # the first activation is a special case -- 
        # it's just input feature vector itself
        A = [np.atleast_2d(x)]
        # A above is the list containing activation functions for the layer
        # which is initialised first by input data points
        
        # FEEDFORWARD
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and 
            # the weight matrix -- this is called the "net input"
            # to the current layer
            net = A[layer].dot(self.W[layer])
            
            # computing the "net output" is simply applying our 
            # nonlinear activation function to the net input
            out = self.sigmoid(net)
            
            # once we have the net output, add it to out list of #
            # activations
            A.append(out)
            # Which acts as the acivation for the next layers
            
            # BACKPROPAGATION
            # the first phase of backpropagation is to compute the 
            # difference between our *prediction* (the final output
            # activation in the activations list) and the true target value
            
            error = A[-1] - y
            
            # from here, we need to apply the chain rule and build our 
            # list of deltas 'D'; the first entry in the deltas is 
            # simply the error of the output layer times the derivative
            # of our activation function for the output value
            D = [error * self.sigmoid_deriv(A[-1])]
            
            # Remember the chain rule works from outside funtion to inside function
            # In terms of layers, from outside layer to inside layer
            # Thus in backpropagation, it starts from the last layer till the second
            # computing all the gradients
            for layer in np.arange(len(A) - 2, 0, -1):
                # the delta for the current layer is equal to the delta of 
                # the *previous layer* dotted withe the weight matrix of 
                # current layer, followed by multiplying the delta by the  
                # derivative of the nonlinear activation function 
                # for the activations of the current layer
                delta = D[-1].dot(self.W[layer].T)
                delta = delta*self.sigmoid_deriv(A[layer])
                D.append(delta)
                
                D = D[::-1] 
                # deltas reversed because we looped the layers in reverse order
                
                # Updating the weights
                for layer in np.arange(0, len(self.W)):
                    # update our weights by taking the dot product 
                    # of layer activations with the respective deltas,
                    # then muliplying this value by some small learning rate and
                    # adding to out weight matrix -- this is where the actual "learning"
                    # takes place
                    self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
                    # This line was the gradient descent
                    
    def predict(self, X, addBias = True):
        # initialise the output as input features 
        # this value will be (forward) propagated through 
        # the network to obtain the final prediction
        p = np.atleast_2d(X)
        
        # check to see if the bias column should be added
        if addBias:
            # inserting 1's column in the last of feature matrix for bias
            p = np.c_[p, np.ones((p.shape[0]))]
            
        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # Computing output = dot product of 
            # current activation function and weight
            # matrix associated with current layer 
            # Then passing this value through a non-linear
            # activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))
            
        # return the predicted value
        return p
    
    # The fianl function of the class is below
    def calculate_loss(self, X, targets):
        # loss = prediction - targets
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        
        return loss
                    
                
        
        
    
if __name__ == '__main__':
    
    # Checking on XOR dataset
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork([2,2,1], alpha = 0.5)
    nn.fit(X, y, epochs = 2000)
    
    for (x,target) in zip(X,y):
        pred = nn.predict(x)[0][0]
        step = 1 if pred > 0.5 else 0
        print("[INFO] data={}, ground truth = {}, pred ={:.4f}, step ={}".format(x, target[0], pred, step))
    
    
    