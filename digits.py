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
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    x = M["train5"][150].reshape((28*28, 1))
    x /= 255

    W = np.random.rand(10, 28*28)
    b = np.random.rand(10, 1)

    return compute_simple_network(x, W, b)

################################################################################
# Part 3
################################################################################
def part3():
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    x = M["train5"][150].reshape((28*28, 1))
    x /= 255

    W = np.random.rand(10, 28*28)
    b = np.random.rand(10, 1)
    y = np.random.rand(10, 1)

    # h value
    h = 0.000001

    check_finite_differences(x, W, b, y, h)

################################################################################
# Part 4
################################################################################
def part4():
    pass
