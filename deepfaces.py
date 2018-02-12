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
# def part8():
# Actors for training and validation set
act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

# Uncomment if images need to be downloaded in ./cropped/ folder
# If it doesn't work, unzip cropped.zip
############################################################################
# get_and_crop_images(act)
# remove_bad_images()
############################################################################

train_set, train_label = np.zeros((0, 64*64)), np.zeros((0, len(act)))
test_set, test_label = np.zeros((0, 64*64)), np.zeros((0, len(act)))

for i in range(len(act)):
    a_name = act[i].split()[1].lower()

    train_set_i, test_set_i = build_sets(a_name)

    one_hot = np.zeros(len(act))
    one_hot[i] = 1

    train_set = np.vstack((train_set, train_set_i))
    test_set = np.vstack((test_set, test_set_i))

    train_label = np.vstack((train_label, np.tile(one_hot, (train_set_i.shape[0], 1))))
    test_label = np.vstack((test_label, np.tile(one_hot, (test_set_i.shape[0], 1))))

dim_x = 64*64
dim_h = 30
dim_out = len(act)

dtype_float = torch.FloatTensor

x = Variable(torch.from_numpy(train_set[:]), requires_grad=False).type(dtype_float)
y = Variable(torch.from_numpy(train_label[:].astype(float)), requires_grad=False).type(dtype_float)


b0 = Variable(torch.randn((1, dim_h)), requires_grad=True)
W0 = Variable(torch.randn((dim_x, dim_h)), requires_grad=True)

b1 = Variable(torch.randn((1, dim_out)), requires_grad=True)
W1 = Variable(torch.randn((dim_h, dim_out)), requires_grad=True)

logSoftMax = torch.nn.LogSoftmax(dim=1)

learning_rate = 9e-1
for t in range(1000):
    y_out = nn_model(x, b0, W0, b1, W1)

    loss = -torch.mean(torch.sum(y * logSoftMax(y_out), 1))
    loss.backward()

    b0.data -= learning_rate * b0.grad.data
    W0.data -= learning_rate * W0.grad.data
    
    b1.data -= learning_rate * b1.grad.data
    W1.data -= learning_rate * W1.grad.data

    b0.grad.data.zero_()
    W0.grad.data.zero_()
    b1.grad.data.zero_()
    W1.grad.data.zero_()

    if t % 100 == 0 or t == 1000 - 1:
        print("Epoch: " + str(t))

        y_pred = nn_model(x, b0, W0, b1, W1).data.numpy()

        print("Training Set Performance: " + str((np.mean(np.argmax(y_pred, 1) == np.argmax(train_label, 1))) * 100) + "%")        

        x_test = Variable(torch.from_numpy(test_set), requires_grad=False).type(dtype_float)

        y_pred = nn_model(x_test, b0, W0, b1, W1).data.numpy()

        print("Testing Set Performance:  " + str((np.mean(np.argmax(y_pred, 1) == np.argmax(test_label, 1))) * 100) + "%\n")

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
# part8()
# part9()
# part10()
