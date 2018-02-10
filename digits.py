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
    x /= 255

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
    x /= 255

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
def part4():
    pass


################################################################################
# Function calls
################################################################################
# part1()
# part2()
# part3()
# part4()