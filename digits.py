from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

#for contour plots
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

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

    #testing whether it's same as previous gradient function
    # grad_mat_W_original = gradient_simple_network_w(x, W, b1, y)
    grad_mat_W_modified = gradient_simple_network_w(x, W, b1, y)
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


    #Saved weights by training for 3000 iterations only in the interest of time
    print("saving")
    np.save("weights_part5", W)
    np.save("bias_part5", b)

    # print("plotting")
    # plot_learning_curves("part5", epoch, train_perf, test_perf)
    # plot_digit_weights(W)

################################################################################
# Part 6
################################################################################
def part6a():

    M = loadmat("mnist_all.mat")

    # Split data into training and test set
    train_set, train_label = np.zeros((0, 28 * 28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28 * 28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(M["train" + str(i)])[:]) / 255.)))
        test_set = np.vstack((test_set, ((np.array(M["test" + str(i)])[:]) / 255.)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(M["train" + str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(M["test" + str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T

    weights = np.load("weights_part5.npy")
    bias = np.load("bias_part5.npy")

    #gradient = np.load("gradient_matrix_part5.npy")
    # np.amax(gradient)
    # Out[81]: 7.5869704674327316
    # np.where(gradient == np.amax(gradient))
    # Out[82]: (array([360]), array([5]))
    # np.where(gradient == np.amin(gradient))
    # Out[83]: (array([333]), array([5]))

    #choose weights at the center of the digits 784/2 and 784/2 + 28* 5
    w1 = weights[360, 5]
    w2 = weights[333, 5]
    # are my weights supposed to be this small?
    #print(w1, w2)

    coords = np.array([[360, 333],[5, 5]])
    # You should determine the range that would get you a good visualization.
#    print(weights)
    #print(np.mean(weights), np.std(weights))
    #print(np.mean(weights[int(784/2),:]), np.std(weights[:,3]), np.std(weights[:,4]))
    #choose 20 values to vary weights by
    #bad idea lol
    # w1_range = np.random.normal(w1, np.std(weights[:,3]), 20)
    # w2_range = np.random.normal(w2, np.std(weights[:,4]), 20)

    w1_range = np.arange(-10, -7, 0.1)
    w2_range = np.arange(2, 5, 0.1)
    #print(w1, w2)

    print("computing cost matrix")
    cost = cost_for_contour(train_set, weights, bias, train_label, w1_range, w2_range, coords)
    print("creating contour plot")
    create_contour_plot(cost, w1_range, w2_range)

def part6b():

    M = loadmat("mnist_all.mat")

    # Split data into training and test set
    train_set, train_label = np.zeros((0, 28 * 28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28 * 28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(M["train" + str(i)])[:]) / 255.)))
        test_set = np.vstack((test_set, ((np.array(M["test" + str(i)])[:]) / 255.)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(M["train" + str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(M["test" + str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T

    weights = np.load("weights_part5.npy")
    bias = np.load("bias_part5.npy")

    w1_coords = (360, 5)
    w2_coords = (333, 5)

    init_W = weights.copy()
    #initialize these weights to be far apart from original values
    # weights = np.load("weights_part5.npy")
    # weights[360, 5]
    # Out[85]: -1.1622765026406856
    # weights[333, 5]
    # Out[86]: 1.2262973718837111
    init_W[w1_coords[0], w1_coords[1]] = -9.8
    init_W[w2_coords[0], w2_coords[1]] = 2.3
    init_b = np.load("bias_part5.npy")

    alpha = 0.01
    weights_progress = train_nn_p6b(compute_simple_network, gradient_simple_network_w,
                                                  gradient_simple_network_b, train_set, train_label, test_set,
                                                  test_label, init_W, init_b, alpha, 20, w1_coords, w2_coords)
    print(weights_progress) #weights seem ok

    w1 = weights[360, 5]
    w2 = weights[333, 5]
    coords = np.array([[360, 333], [5, 5]])
    w1_range = np.arange(-10, -7, 0.1)
    w2_range = np.arange(2, 5, 0.1)

    cost = cost_for_contour(train_set, weights, bias, train_label, w1_range, w2_range, coords)
    plot_trajectories(cost, w1_range, w2_range, weights_progress)

def part6c():
    M = loadmat("mnist_all.mat")

    # Split data into training and test set
    train_set, train_label = np.zeros((0, 28 * 28)), np.zeros((0, 10))
    test_set, test_label = np.zeros((0, 28 * 28)), np.zeros((0, 10))

    for i in range(10):
        train_set = np.vstack((train_set, ((np.array(M["train" + str(i)])[:]) / 255.)))
        test_set = np.vstack((test_set, ((np.array(M["test" + str(i)])[:]) / 255.)))

        one_hot = np.zeros(10)
        one_hot[i] = 1

        train_label = np.vstack((train_label, np.tile(one_hot, (len(M["train" + str(i)]), 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (len(M["test" + str(i)]), 1))))

    train_set, train_label, test_set, test_label = train_set.T, train_label.T, test_set.T, test_label.T

    weights = np.load("weights_part5.npy")
    bias = np.load("bias_part5.npy")

    w1_coords = (360, 5)
    w2_coords = (333, 5)

    init_W = weights.copy()
    #initialize these weights to be far apart from original values
    # weights = np.load("weights_part5.npy")
    # weights[360, 5]
    # Out[85]: -1.1622765026406856
    # weights[333, 5]
    # Out[86]: 1.2262973718837111
    init_W[w1_coords[0], w1_coords[1]] = -9.8
    init_W[w2_coords[0], w2_coords[1]] = 2.3
    init_b = np.load("bias_part5.npy")

    alpha = 0.01
    gamma = 0.9
    max_iter = 20

    weights_progress = train_nn_M_p6c(compute_simple_network, gradient_simple_network_w, gradient_simple_network_b, train_set, train_label, test_set, test_label, init_W, init_b, alpha, gamma, max_iter,w1_coords, w2_coords)
    print(weights_progress) #weights seem ok

    # w1 = weights[360, 5]
    # w2 = weights[333, 5]
    # coords = np.array([[360, 333], [5, 5]])
    # w1_range = np.arange(-10, -7, 0.1)
    # w2_range = np.arange(2, 5, 0.1)
    #
    # cost = cost_for_contour(train_set, weights, bias, train_label, w1_range, w2_range, coords)
    # plot_trajectories(cost, w1_range, w2_range, weights_progress)


################################################################################
# Function calls
################################################################################
#part1()
#part2()
#part3()
#part4()
#part5()
#part6a()
#part6b()
part6c()

