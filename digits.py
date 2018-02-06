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

from mnist_handouts import *
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

    x = M["train5"][150].reshape((784, 1))
    x /= 255

    W = np.random.rand(10, 784)
    b = np.random.rand(10, 1)

    return compute_simple_network(x, W, b)

################################################################################
# Part 3
################################################################################
def part3():
    pass