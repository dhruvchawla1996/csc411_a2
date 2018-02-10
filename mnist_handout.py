from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

################################################################################
# #Load the MNIST digit data
# M = loadmat("mnist_all.mat")

# #Display the 150-th "5" digit from the training set
# imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
# show()
################################################################################


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    
def compute_simple_network(x, W, b):
    '''Compute a simple network (with no hidden layers)
    '''
    o = np.dot(W.T, x) + b
    return softmax(o)

def gradient_simple_network_w(x, W, b, y):
    p = compute_simple_network(x, W, b)

    gradient_mat = np.zeros((28*28, 10))

    for j in range(28*28):
        gradient_mat[j, :] = np.dot((p - y), x[j, :].T)

    return gradient_mat

def gradient_simple_network_b(x, W, b, y):
    p = compute_simple_network(x, W, b)

    return (p - y)

def check_finite_differences_w(x, W, b, y, h):
    for j in range(28*28):
        for i in range(10):
            h_mat = np.zeros((28*28, 10))
            h_mat[j, i] = h

            actual_grad = gradient_simple_network_w(x, W, b, y)[j, i]

            cost_fn = NLL(compute_simple_network(x, W, b), y)
            cost_fn_h = NLL(compute_simple_network(x, W+h_mat, b), y)

            finite_diff_grad = (cost_fn_h - cost_fn)/h
            
            if actual_grad != 0:
                print("Index: " + str(j) + ", " + str(i))
                print("Actual Gradient Value: " + str(actual_grad))
                print("Finite Difference Value: " + str(finite_diff_grad) + "\n")

def check_finite_differences_b(x, W, b, y, h):
    for i in range(10):
        h_mat = np.zeros((10, 1))
        h_mat[i] = h

        actual_grad = gradient_simple_network_b(x, W, b, y)[i, 0]

        cost_fn = NLL(compute_simple_network(x, W, b), y)
        cost_fn_h = NLL(compute_simple_network(x, W, b+h_mat), y)

        finite_diff_grad = (cost_fn_h - cost_fn)/h

        if actual_grad != 0:
            print("Index: " + str(i))
            print("Actual Gradient Value: " + str(actual_grad))
            print("Finite Difference Value: " + str(finite_diff_grad) + "\n")

################################################################################
# #Load sample weights for the multilayer neural network
# snapshot = cPickle.load(open("snapshot50.pkl"))
# W0 = snapshot["W0"]
# b0 = snapshot["b0"].reshape((300,1))
# W1 = snapshot["W1"]
# b1 = snapshot["b1"].reshape((10,1))

# #Load one example from the training set, and run it through the
# #neural network
# x = M["train5"][148:149].T    
# L0, L1, output = forward(x, W0, b0, W1, b1)
# #get the index at which the output is the largest
# y = argmax(output)
################################################################################

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
