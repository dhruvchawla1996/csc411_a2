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
    #TODO: push this change, random.seed
    np.random.seed(5)
    y = np.random.rand(10, 1)
    y /= sum(y)

    # h value
    h = 0.001


    grad_mat = gradient_simple_network_b(x, W, b1, y)
    print(grad_mat.shape)
    # print("Checking Difference with respect to weights: \n")
    # check_finite_differences_w(x, W, b1, y, h)
    #
    # print("\nChecking Difference with respect to biases: \n")
    # check_finite_differences_b(x, W, b1, y, h)


def part3_rerun():
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
    #TODO: push this change, random.seed
    np.random.seed(5)
    y = np.random.rand(10, 1)
    y /= sum(y)

    # h value
    h = 0.001

    #testing whether it's same as previous gradient function
    # grad_mat_W_original = gradient_simple_network_w(x, W, b1, y)
    grad_mat_W_modified = gradient_simple_network_w_2(x, W, b1, y)
    # print(grad_mat_W_original.shape, grad_mat_W_modified.shape)
    # print(grad_mat_W_modified == grad_mat_W_original)

    #TODO: choose nonzero coordinates
    coords = np.zeros((2, 50))
    np.random.seed(5000)
    random_is = np.random.randint(0, 784, (1, coords.shape[1]))
    np.random.seed(5001)
    random_js = np.random.randint(0, 10, (1, coords.shape[1]))
    for i in range(coords.shape[1]):
        random_i = random_is[0][i]
        random_j = random_js[0][i]
        coords[:, i] = np.array([random_i, random_j])

    # for i in range(coords.shape[1]):
    #     print(grad_mat_W_modified[int(coords[0,i]), int(coords[1,i])])

    check_grad_w(x, W, b1, y, h, coords)

    # print("Checking Difference with respect to weights: \n")
    # check_finite_differences_w(x, W, b1, y, h)
    #
    # print("\nChecking Difference with respect to biases: \n")
    # check_finite_differences_b(x, W, b1, y, h)

################################################################################
# Part 4
################################################################################
def part4():
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    # Split data into training and test set
    train_set, train_label = np.zeros((0, 28*28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28*28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(M["train"+str(i)])[:])/255.)))
        test_set = np.vstack((test_set, ((np.array(M["test"+str(i)])[:])/255.)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(M["train"+str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(M["test"+str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T

    # Load weights and biases
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    init_W = np.dot(W0, W1)
    init_b = b1

    alpha = 0.00001

    W, b, epoch, train_perf, test_perf = train_nn(compute_simple_network, gradient_simple_network_w, gradient_simple_network_b, train_set, train_label, test_set, test_label, init_W, init_b, alpha)

    plot_learning_curves("part4", epoch, train_perf, test_perf)
    plot_digit_weights(W)


def part4_rerun():
     #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    # Split data into training and test set
    train_set, train_label = np.zeros((0, 28*28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28*28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(M["train"+str(i)])[:])/255.)))
        test_set = np.vstack((test_set, ((np.array(M["test"+str(i)])[:])/255.)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(M["train"+str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(M["test"+str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T

    # Load weights and biases
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    init_W = np.dot(W0, W1)
    init_b = b1

    alpha = 0.00001

    W, b, epoch, train_perf, test_perf = train_nn(compute_simple_network, gradient_simple_network_w_2, gradient_simple_network_b, train_set, train_label, test_set, test_label, init_W, init_b, alpha)

    plot_learning_curves_rerun("part4", epoch, train_perf, test_perf)
    plot_digit_weights_rerun(W)

################################################################################
# Part 5
################################################################################
def part5():
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    # Split data into training and test set
    train_set, train_label = np.zeros((0, 28*28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28*28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(M["train"+str(i)])[:])/255.)))
        test_set = np.vstack((test_set, ((np.array(M["test"+str(i)])[:])/255.)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(M["train"+str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(M["test"+str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T

    # Load weights and biases
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    init_W = np.dot(W0, W1)
    init_b = b1

    alpha = 0.00001

    W, b, epoch, train_perf, test_perf = train_nn_M(compute_simple_network, gradient_simple_network_w, gradient_simple_network_b, train_set, train_label, test_set, test_label, init_W, init_b, alpha)

    print("plotting")
    plot_learning_curves("part5", epoch,train_perf, test_perf)
    plot_digit_weights(W)


def part5_rerun():
    '''
    #TODO test part_5 after testing part4 and checking parameter update for momentum
    '''

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    # Split data into training and test set
    train_set, train_label = np.zeros((0, 28*28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28*28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(M["train"+str(i)])[:])/255.)))
        test_set = np.vstack((test_set, ((np.array(M["test"+str(i)])[:])/255.)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(M["train"+str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(M["test"+str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T

    # Load weights and biases
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    init_W = np.dot(W0, W1)
    init_b = b1

    alpha = 0.00001

    W, b, epoch, train_perf, test_perf = train_nn_M2(compute_simple_network, gradient_simple_network_w_2, gradient_simple_network_b, train_set, train_label, test_set, test_label, init_W, init_b, alpha)

    print("plotting")
    plot_learning_curves_rerun("part5", epoch,train_perf, test_perf)
    plot_digit_weights_rerun(W)

################################################################################
# Part 6
################################################################################
def part6():

    plot_trajectory("part6", mo_traj, w1, w2)
    

################################################################################
# Function calls
################################################################################
#part1()
#part2()
#part3_rerun()
#part4_rerun()
part5_rerun()
# part6()
