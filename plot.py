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

plt.switch_backend('agg')

def plot_each_digit():
    """Save 10 images of each of the 10 digits in the dataset
    """

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    f, axarr = plt.subplots(10, 10)

    for i in range(10):
        for j in range(10):
            axarr[i][j].imshow(M["train"+str(i)][j].reshape((28,28)), cmap=cm.gray)
            axarr[i][j].axis('off')

    plt.savefig("part1.png")
