SEED=123

import os
os.environ['HOME'] = './' #'/root'
#os.environ['PYTHONUNBUFFERED'] = '1'

SIZE = 160 #180
BASE_DIR = '/home/cory/Kaggle/Cdiscount/'
DATA_DIR = BASE_DIR+'input/data/'
OUT_DIR = BASE_DIR+'resnet50-pretrain/' #'/densenet201-pretrain-288/' 
AUX_DIR = '/media/cory/bb09faa5-afbf-46eb-ad07-c0252e84e93b/Kaggle/Cdiscount/input/'

#numerical libs
import math
import numpy as np
import random
import PIL
import cv2

random.seed(SEED)
np.random.seed(SEED)

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg')



# torch libs
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torch.backends.cudnn as cudnn
cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -



# std libs
import inspect
from shutil import copyfile
import pickle
from timeit import default_timer as timer   #ubuntu:  default_timer = time.time,  seconds
from datetime import datetime
import csv
import pandas as pd
import pickle
import glob
import sys
#from time import sleep

import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics

from skimage import io

#import io
import bson                       # this is installed with the pymongo package
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data


'''
updating pytorch
    https://discuss.pytorch.org/t/updating-pytorch/309
    
    ./conda config --add channels soumith

'''
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        #labels = labels.float()
        #output= net(Variable(images.cuda(),volatile=True))
        logits, probs = net(Variable(images.cuda()))
        loss  = criterion(logits, labels.cuda())

        #_, predictions = torch.max(probs, 1)

        batch_size = len(images)
        test_acc  += batch_size*acc_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == test_loader.dataset.num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc

def evaluate_and_predict(net, test_loader, num_classes):

    test_dataset = test_loader.dataset
    #num_classes  = len(test_dataset.class_names)
    predictions  = np.zeros((test_dataset.num,num_classes),np.uint8) #np.float32)

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))
        loss  = criterion(logits, labels.cuda())
	#multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*acc_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1,num_classes)

    assert(test_dataset.num==test_num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc, predictions

def acc_measure(predictions, labels, threshold=0.235):
    batch_size = predictions.size()[0] # will it be the same?
    # or len(p)??
    #print('predictions', predictions)
    l = labels
    p = torch.max(predictions, 1)[1]
    #print('p', p)
    #print('l', l)
    #p = probs.apply_(lambda x: torch.max(x, 1))

    num_correct = sum(list(map(lambda z: z[0] == z[1], zip(p.cpu().numpy(), l.cpu().numpy()))))
    #for x, y in zip(p.data, l.data)

    acc = num_correct/batch_size

    return acc

# loss ----------------------------------------
def criterion(logits, labels):
    #print('logits:', logits)
    #print('lables:', labels)
    loss = nn.CrossEntropyLoss()(logits, Variable(labels))
    return loss

