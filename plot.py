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


def plot_trajectories(cost, w1, w2, weights_progress, part):
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    W1, W2 = np.meshgrid(w1, w2)

    plt.figure()
    CS = plt.contour(W1, W2, cost)

    plt.plot([a for a, b in weights_progress], [b for a, b in weights_progress], 'yo-', label="No Momentum")

    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.title('Contour plot with trajectory (no momentum)')
    plt.legend(loc="best")
    plt.savefig("figures/part6"+part+".png")

def plot_trajectories_M(cost, w1, w2, weights_progress, part):
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    W1, W2 = np.meshgrid(w1, w2)

    plt.figure()
    CS = plt.contour(W1, W2, cost)

    plt.plot([a for a, b in weights_progress], [b for a, b in weights_progress], 'yo-', label="With Momentum")

    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.title('Contour plot with trajectory (With momentum)')
    plt.legend(loc="best")
    plt.savefig("figures/part6"+part+".png")

# def plot_both_trajectories(cost, w1, w2, weights_progress, weights_progress_M, part):
#     matplotlib.rcParams['xtick.direction'] = 'out'
#     matplotlib.rcParams['ytick.direction'] = 'out'
#
#     W1, W2 = np.meshgrid(w1, w2)
#
#     plt.figure()
#     CS = plt.contour(W1, W2, cost)
#
#     plt.plot([a for a, b in weights_progress], [b for a, b in weights_progress], 'yo-', label="No Momentum")
#     plt.plot([a for a, b in weights_progress_M], [b for a, b in weights_progress], 'yo-', label="With Momentum")
#     plt.xlabel("w1")
#     plt.ylabel("w2")
#     plt.title('Contour plot with trajectories comparing gradient descent with and without momentum')
#     plt.legend(loc="best")
#     plt.savefig("figures/part6"+part+".png")

# def plot_trajectories(cost, w1, w2):
#
#     gd_traj = [(init_w1, init_w2), (step1_w1, step1_w2), ...]
#     mo_traj = [(init_w1, init_w2), (step1_w1, step1_w2), ...]
#     w1s = np.arange(-0, 1, 0.05)
#     w2s = np.arange(-0, 1, 0.05)
#     w1z, w2z = np.meshgrid(w1s, w2s)
#     C = np.zeros([w1s.size, w2s.size])
#     for i, w1 in enumerate(w1s):
#         for j, w2 in enumerate(w2s):
#             C[i,j] = get_loss(w1, w2)
#     CS = plt.contour(w1z, w2z, C)
#     plt.plot([a for a, b in gd_traj], [b for a,b in gd_traj], 'yo-', label="No Momentum")
#     plt.plot([a for a, b in mo_traj], [b for a,b in mo_traj], 'go-', label="Momentum")
#     plt.legend(loc='top left')
#     plt.title('Contour plot')


#def plot_trajectory_Sabrina(part, w1, w2)
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

