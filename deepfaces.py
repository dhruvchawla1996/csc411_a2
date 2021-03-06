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
from torch.autograd import Variable
import torch

from mnist_handout import *
from plot import *
from faces import *
from myalexnet import *

################################################################################
# Part 8
################################################################################
def part8():
    # Actors for training and validation set
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    # Uncomment if images need to be downloaded in ./cropped/ folder
    # If it doesn't work, unzip cropped.zip
    ############################################################################
    # get_and_crop_images(act, 64)
    # remove_bad_images(64)
    ############################################################################

    train_set, train_label = np.zeros((0, 64*64*3)), np.zeros((0, len(act)))
    test_set, test_label = np.zeros((0, 64*64*3)), np.zeros((0, len(act)))

    for i in range(len(act)):
        a_name = act[i].split()[1].lower()

        train_set_i, test_set_i = build_sets_part8(a_name)

        one_hot = np.zeros(len(act))
        one_hot[i] = 1

        train_set = np.vstack((train_set, train_set_i))
        test_set = np.vstack((test_set, test_set_i))

        train_label = np.vstack((train_label, np.tile(one_hot, (train_set_i.shape[0], 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (test_set_i.shape[0], 1))))

    # Set seed value for generating initial weights
    torch.manual_seed(5)

    dim_x = 64*64*3
    dim_h = 600
    dim_out = len(act)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    # Using mini-batches for training
    np.random.seed(5)
    train_idx = np.random.permutation(range(train_set.shape[0]))[:600]

    x = Variable(torch.from_numpy(train_set[train_idx]), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_label[train_idx], 1)), requires_grad=False).type(dtype_long)

    mini_batch_size = 20

    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch, train_perf, test_perf = [], [], []

    learning_rate, max_iter = 1e-4, 700
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(max_iter+1):
        y_pred = model(x[(t*mini_batch_size)%600:(t*mini_batch_size)%600 + mini_batch_size])
        loss = loss_fn(y_pred, y_classes[(t*mini_batch_size)%600:(t*mini_batch_size)%600 + mini_batch_size])
        
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                           # make a step

        if t % 100 == 0:
            print("Epoch: " + str(t))

            # Training Performance
            x_train = Variable(torch.from_numpy(train_set[:]), requires_grad=False).type(dtype_float)
            y_pred = model(x_train).data.numpy()
            train_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(train_label, 1))) * 100
            print("Training Set Performance: " + str(train_perf_i) + "%")      

            # Testing Performance  

            x_test = Variable(torch.from_numpy(test_set), requires_grad=False).type(dtype_float)
            y_pred = model(x_test).data.numpy()
            test_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(test_label, 1))) * 100
            print("Testing Set Performance:  " + str(test_perf_i) + "%\n")

            epoch.append(t)
            train_perf.append(train_perf_i)
            test_perf.append(test_perf_i)

    plot_learning_curves("part8", epoch, train_perf, test_perf)

    # Save first layer's parameters in a pickle file (for part 9)
    model_params = {}
    model_params["W"] = model[0].weight.data.numpy()
    model_params["b"] = model[0].bias.data.numpy()

    cPickle.dump(model_params, open("part8_model_params.pkl", "wb"))

################################################################################
# Part 9
################################################################################
def part9():
    # Remove folders figures/bracco/ and figures/baldwin
    if os.path.exists('./figures/bracco'): shutil.rmtree('./figures/bracco')
    if os.path.exists('./figures/baldwin'): shutil.rmtree('./figures/baldwin')

    # Create figures/bracoo and figures/baldwin
    if not os.path.exists('./figures/bracco'): os.makedirs('./figures/bracco')
    if not os.path.exists('./figures/baldwin'): os.makedirs('./figures/baldwin')

    # Load weights from the model of part8
    snapshot = cPickle.load(open("part8_model_params.pkl", "rb"))
    W = snapshot["W"]
    b = snapshot["b"]
    b = b.reshape((b.shape[0], 1))

    # Let's open an image for Bracco and see which hidden neurons are firing more
    img = imread("cropped64/bracco27.jpg")
    img = img[:, :, :3]
    img = reshape(np.ndarray.flatten(img), [1, 64*64*3])
    img = img/128. - 1.

    h = myReLU(np.dot(W, img.T) + b)
    h = softmax(h)
    h_max_i = []

    # Get 10 most active neuron's indices
    for i in range(10):
        h_max_i.append(np.argmax(h))
        h[h_max_i[-1]] = 0
        
    ctr = 0
    for i in h_max_i:
        W_i = W[i, :].reshape((64, 64, 3))
        W_i = (W_i[:,:,0] + W_i[:,:,1] + W_i[:,:,2])/255.
        imsave("figures/bracco/part9_bracco_"+str(ctr)+".jpg", W_i, cmap = "RdBu")
        ctr = ctr + 1

    # Let's open an image for Baldwin and see which hidden neurons are firing more
    img = imread("cropped64/baldwin38.jpg")
    img = img[:, :, :3]
    img = reshape(np.ndarray.flatten(img), [1, 64*64*3])
    img = img/128. - 1.

    h = myReLU(np.dot(W, img.T) + b)
    h = softmax(h)
    h_max_i = []

    # Get 10 most active neuron's indices
    for i in range(10):
        h_max_i.append(np.argmax(h))
        h[h_max_i[-1]] = 0
        
    ctr = 0
    for i in h_max_i:
        W_i = W[i, :].reshape((64, 64, 3))
        W_i = (W_i[:,:,0] + W_i[:,:,1] + W_i[:,:,2])/255.
        imsave("figures/baldwin/part9_baldwin_"+str(ctr)+".jpg", W_i, cmap = "RdBu")
        ctr = ctr + 1

################################################################################
# Part 10
################################################################################
def part10():
    # Actors for training and validation set
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    # Uncomment if images need to be downloaded in ./cropped/ folder
    # If it doesn't work, unzip cropped.zip
    ############################################################################
    # get_and_crop_images(act, 227)
    #remove_bad_images(227)
    ############################################################################

    alexNetFaceScrub()

################################################################################
# Function calls
################################################################################
# part8()
# part9()
# part10()
