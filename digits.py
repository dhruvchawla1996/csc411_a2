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

from mnist_handout import *
from plot import *

################################################################################
# Part 1
################################################################################
def part1():
    plot_each_digit()

################################################################################
# Part 2
################################################################################
def part2():
    np.random.seed(5)

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    x = M["train5"][150].reshape((28*28, 1))
    x = x/255.0

    W = np.random.rand(28*28, 10)
    b = np.random.rand(10, 1)

    return compute_simple_network(x, W, b)

################################################################################
# Part 3
################################################################################
def part3():
    np.random.seed(5)

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    # Load an image
    x = M["train5"][150].reshape((28*28, 1))
    x = x/255.0

    # Load weights and biases
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    W = np.dot(W0, W1)

    # Construct a sample output (all elements sum up to unity)
    y = np.random.rand(10, 1)
    y /= sum(y)

    # h value
    h = 0.001


    print("Checking Difference with respect to weights: \n")
    check_finite_differences_w(x, W, b1, y, h)

    print("\nChecking Difference with respect to biases: \n")
    check_finite_differences_b(x, W, b1, y, h)

################################################################################
# Part 4
################################################################################
# def part4():
#Load the MNIST digit data
M = loadmat("mnist_all.mat")

# Set sizes for training, validation and testing sets for each digit
training_size = 400
validation_size = 50
test_size = 50

# Make the three sets
train_label = np.zeros((10, training_size * 10))
validation_label = np.zeros((10, validation_size * 10))
test_label = np.zeros((10, test_size * 10))

# For digit '0'
training_set = M["train0"][0:training_size]
validation_set = M["train0"][training_size:training_size+validation_size]
test_set = M["train0"][training_size+validation_size:training_size+validation_size+test_size]

train_label[0, 0:training_size] = 1
validation_label[0, 0:validation_size] = 1
test_label[0, 0:test_size] = 1

# For the rest of the digits
for i in range(1, 10):
    training_set = vstack((training_set, M["train"+str(i)][0:training_size]))
    validation_set = vstack((validation_set, M["train"+str(i)][training_size:training_size+validation_size]))
    test_set = vstack((test_set, M["train"+str(i)][training_size+validation_size:training_size+validation_size+test_size]))

    train_label[i, i*training_size:i*training_size+training_size] = 1
    validation_label[i, i*validation_size:i*validation_size+validation_size] = 1
    test_label[i, i*test_size:i*test_size+test_size] = 1

training_set = training_set.T
validation_set = validation_set.T
test_set = test_set.T

training_set = training_set/255.0
validation_set = validation_set/255.0
test_set = test_set/255.0

# Load weights and biases
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

init_W = np.dot(W0, W1)
init_b = b1

alpha = 0.0001

W, b, epoch, train_perf, validation_perf, test_perf = train_nn(compute_simple_network, gradient_simple_network_w, gradient_simple_network_b, training_set, train_label, validation_set, validation_label, test_set, test_label, init_W, init_b, alpha)

plot_learning_curves(epoch, train_perf, validation_perf, test_perf)
plot_digit_weights(W)

################################################################################
# Part 5
################################################################################
def part5():
    pass

################################################################################
# Part 6
################################################################################
def part6():
    pass

################################################################################
# Function calls
################################################################################
# part1()
# part2()
# part3()
# part4()
# part5()
# part6()
