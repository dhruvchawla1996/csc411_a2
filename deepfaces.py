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

################################################################################
# Part 8
################################################################################
def part8():
    # Actors for training and validation set
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    # Uncomment if images need to be downloaded in ./cropped/ folder
    # If it doesn't work, unzip cropped.zip
    ############################################################################
    # get_and_crop_images(act)
    # remove_bad_images()
    ############################################################################

    train_set, train_label = np.zeros((0, 64*64*3)), np.zeros((0, len(act)))
    test_set, test_label = np.zeros((0, 64*64*3)), np.zeros((0, len(act)))

    for i in range(len(act)):
        a_name = act[i].split()[1].lower()

        train_set_i, test_set_i = build_sets(a_name)

        one_hot = np.zeros(len(act))
        one_hot[i] = 1

        train_set = np.vstack((train_set, train_set_i))
        test_set = np.vstack((test_set, test_set_i))

        train_label = np.vstack((train_label, np.tile(one_hot, (train_set_i.shape[0], 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (test_set_i.shape[0], 1))))

    # Set seed value for generating initial weights
    torch.manual_seed(5)

    dim_x = 64*64*3
    dim_h0 = 250
    dim_h1 = 40
    dim_out = len(act)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    # Using mini-batches for training
    np.random.seed(5)
    train_idx = np.random.permutation(range(train_set.shape[0]))[:600]

    x = Variable(torch.from_numpy(train_set[train_idx]), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(train_label[train_idx], 1)), requires_grad=False).type(dtype_long)


    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h0),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h0, dim_h1),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h1, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch, train_perf, test_perf = [], [], []

    learning_rate, max_iter = 1e-4, 200
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(max_iter):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_classes)
        
        model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                           # make a step

        if t % 20 == 0 or t == max_iter - 1:
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

################################################################################
# Part 9
################################################################################
def part9():
    pass

################################################################################
# Part 10
################################################################################
def part10():
    pass

################################################################################
# Function calls
################################################################################
part8()
# part9()
# part10()
