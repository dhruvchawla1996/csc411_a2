# Imports
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize

import torch.nn as nn

import cPickle

from faces import *

# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 13 * 13)
        # x = self.classifier(x)
        return x

def alexNetFaceScrub():
    ################################################################################
    # Getting Activations from AlexNet
    ################################################################################

    # model_orig = torchvision.models.alexnet(pretrained=True)
    model = MyAlexNet()
    model.eval()

    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

    train_set, train_label = np.zeros((0, 3, 227, 227)), np.zeros((0, len(act)))
    test_set, test_label = np.zeros((0, 3, 227, 227)), np.zeros((0, len(act)))

    for i in range(len(act)):
        a_name = act[i].split()[1].lower()

        train_set_i, test_set_i = build_sets_part10(a_name)

        one_hot = np.zeros(len(act))
        one_hot[i] = 1

        train_set = np.vstack((train_set, train_set_i))
        test_set = np.vstack((test_set, test_set_i))

        train_label = np.vstack((train_label, np.tile(one_hot, (train_set_i.shape[0], 1))))
        test_label = np.vstack((test_label, np.tile(one_hot, (test_set_i.shape[0], 1))))

    train_activation = np.zeros((0, 256*13*13))
    test_activation = np.zeros((0, 256*13*13))

    x = Variable(torch.from_numpy(train_set[:100]), requires_grad=False).type(torch.FloatTensor)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(train_set[100:200]), requires_grad=False).type(torch.FloatTensor)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(train_set[200:300]), requires_grad=False).type(torch.FloatTensor)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(train_set[300:400]), requires_grad=False).type(torch.FloatTensor)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(train_set[400:500]), requires_grad=False).type(torch.FloatTensor)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(train_set[500:600]), requires_grad=False).type(torch.FloatTensor)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(train_set[600:]), requires_grad=False).type(torch.FloatTensor)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(test_set), requires_grad=False).type(torch.FloatTensor)
    test_activation = np.vstack((test_activation, model.forward(x).data.numpy()))

    ################################################################################
    # Using activations as input to Part8 Neural Net
    ################################################################################

    torch.manual_seed(5)

    dim_x = 256*13*13
    dim_h = 600
    dim_out = len(act)

    x = Variable(torch.from_numpy(train_activation), requires_grad=False).type(torch.FloatTensor)
    y_classes = Variable(torch.from_numpy(np.argmax(train_label, 1)), requires_grad=False).type(torch.LongTensor)

    new_model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch, train_perf, test_perf = [], [], []

    learning_rate, max_iter = 1e-4, 150
    optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)
    for t in range(max_iter):
        y_pred = new_model(x)
        loss = loss_fn(y_pred, y_classes)
        
        new_model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                           # make a step

        if t % 10 == 0 or t == max_iter - 1:
            print("Epoch: " + str(t))

            # Training Performance
            x_train = Variable(torch.from_numpy(train_activation), requires_grad=False).type(torch.FloatTensor)
            y_pred = new_model(x_train).data.numpy()
            train_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(train_label, 1))) * 100
            print("Training Set Performance: " + str(train_perf_i) + "%")      

            # Testing Performance  
            x_test = Variable(torch.from_numpy(test_activation), requires_grad=False).type(torch.FloatTensor)
            y_pred = new_model(x_test).data.numpy()
            test_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(test_label, 1))) * 100
            print("Testing Set Performance:  " + str(test_perf_i) + "%\n")

            epoch.append(t)
            train_perf.append(train_perf_i)
            test_perf.append(test_perf_i)