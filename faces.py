# Imports
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import hashlib
import shutil
from torch.autograd import Variable
import torch

################################################################################
# Downloading and populating images from the dataset
################################################################################
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    Manages download by aborting download if it's taking too long'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


def get_and_crop_images(act):
    '''Downloads images from faces_subset.txt
    Stores raw images into ./uncropped and processes them into 32x32 grayscale images in ./cropped

    Requires: faces_subset.txt as obtained from the FaceScrub dataset
                act: list of actor names

    Stores images as lastname_i.jpg. Ex: giplin1.jpg
    '''
    # Remove folders cropped/ and uncropped/
    if os.path.exists('./cropped'): shutil.rmtree('./cropped')
    if os.path.exists('./uncropped'): shutil.rmtree('./uncropped')

    # Create cropped/ and uncropped/
    if not os.path.exists('./cropped'): os.makedirs('cropped')
    if not os.path.exists('./uncropped'): os.makedirs('uncropped')

    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("faces_subset.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 45)

                crop_bbox = line.split()[5].split(',')
                sha256_hash = line.split()[6]

                # Convert uncropped image to cropped 32x32 grayscale
                if not os.path.isfile("uncropped/"+filename):
                    continue

                try:
                    rgb_img = imread("uncropped/"+filename)
                    cropped_img = rgb_img[int(crop_bbox[1]):int(crop_bbox[3]), int(crop_bbox[0]):int(crop_bbox[2])]
                    resized_img = imresize(cropped_img, (64, 64))
                    imsave("cropped/"+filename, resized_img)

                except Exception as e:
                    print(str(e))

                print filename
                i += 1

def remove_bad_images():
    '''Removes bad images from list (manually chosen)
    Requires: cropped images in ./cropped
    '''
    bad_image_filenames = ["baldwin68",
                            "bracco90",
                            "bracco64",
                            "butler131",
                            "butler132",
                            "carell93",
                            "chenoweth29",
                            "chenoweth80",
                            "chenoweth94",
                            "drescher106",
                            "drescher109",
                            "drescher124",
                            "drescher88",
                            "ferrera159",
                            "ferrera123",
                            "hader4",
                            "hader63",
                            "hader77",
                            "hader98",
                            "harmon50",
                            "radcliffe28",
                            "vartan59",
                            "vartan72"]

    for filename in bad_image_filenames:
        if os.path.isfile("cropped/"+filename+".jpg"):
            os.remove("cropped/"+filename+".jpg")

################################################################################
# Building training, validation and testing sets for an actor
################################################################################
def build_sets(actor):
    '''Return two lists of randomized image names 
    in cropped/ folder that match actor name
    
    Training Set - At least 67 image names (Screw Peri Gilpin)
    Validation Set - 10 image names 
    Testing Set - 10 image names
    
    Takes in name as lowercase last name (ex: gilpin)

    Assumption: cropped/ folder is populated with images from get_and_crop_images
    '''
    # Make a list of images for the actor
    image_list = []

    for f in os.listdir("cropped"):
        if actor in f:
            image_list.append(f)

    # Shuffle
    np.random.seed(5)
    np.random.shuffle(image_list)

    train_set = np.zeros((0, 64*64*3))
    test_set = np.zeros((0, 64*64*3))

    for img in image_list[:20]:
        t_img = imread("cropped/"+img)
        t_img = t_img[:, :, :3]
        t_img = reshape(np.ndarray.flatten(t_img), [1, 64*64*3])
        t_img = t_img/128. - 1.
        test_set = np.vstack((test_set, t_img))

    for img in image_list[20:]:
        tr_img = imread("cropped/"+img)
        tr_img = tr_img[:, :, :3]
        tr_img = reshape(np.ndarray.flatten(tr_img), [1, 64*64*3])
        tr_img = tr_img/128. - 1.
        train_set = np.vstack((train_set, tr_img))

    return train_set, test_set

################################################################################
# Convert RGB images to grayscale 
################################################################################
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
