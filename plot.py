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
    '''Save 10 images of each of the 10 digits in the dataset
    '''

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    f, axarr = plt.subplots(10, 10)

    for i in range(10):
        for j in range(10):
            axarr[i][j].imshow(M["train"+str(i)][j].reshape((28,28)), cmap=cm.gray)
            axarr[i][j].axis('off')

    plt.savefig("figures/part1.png")

def plot_digit_weights(W):
    '''Plot heatmap images of digits 0-9

    W   Array of size 28*28

    Saves 10 figures in figures/
    '''
    for i in range(10):
        fig = figure(i)
        ax = fig.gca()
        heatmap = ax.imshow(W[:, i].reshape((28,28)), cmap = cm.coolwarm)    
        fig.colorbar(heatmap, shrink = 0.5, aspect=5)
        savefig("figures/part4_"+str(i)+".png")

def plot_learning_curves(part, epoch, train_perf, test_perf):
    '''Plot learning curves for training and testing set w.r.t epoch
    
    part                    "part4" or "part8"
    epoch       list(Int)   epoch
    train_perf  list(Int)   performance on training set in % with each element corresponding to epoch
    test_perf   list(Int)   performance on testing set in % with each element corresponding to epoch

    Plots and saves figure in "figure/part4_learning_curve.png" or "figure/part8_learning_curve.png"
    '''
    plt.plot(epoch, train_perf, color='k', linewidth=2, marker="o", label="Training Set")
    plt.plot(epoch, test_perf, color='r', linewidth=2, marker="o", label="Testing Set")

    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Performance (%)")
    plt.legend()
    plt.savefig("figures/" + part + "_learning_curve.png")

#def plot_trajectory(part, w1, w2)
#    gd_traj = [(init_w1, init_w2), (step1_w1, step1_w2), ...]
#    mo_traj = [(init_w1, init_w2), (step1_w1, step1_w2), ...]
#    w1 = np.arange(-0, 1, 0.05)
#    w2 = np.arange(-0, 1, 0.05)
#    w1z, w2z = np.meshgrid(w1s, w2s)
#    C = np.zeros([w1s.size, w2s.size])
#    for i, w1 in enumerate(w1s):
#        for j, w2 in enumerate(w2s):
#            C[j,i] = get_loss(w1, w
#    CS = plt.contour(w1z, w2z, C, camp=cm.coolwarm)
#    plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
#    plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
#    plt.legend(loc='top left')
#    plt.title('Contour plot')
#    plt.savefig("figures/" + part + "_learning_curve.png")

