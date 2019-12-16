# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:28:02 2018

@author: Ankita
"""

M = np.float32([[1,0,1],[0,1,0]])
dst = cv2.warpAffine(x_train[0], M,(img_rows, img_cols))


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm 
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras import regularizers

import cv2

img_rows = 16
img_cols = 16
num_classes = 10
batch_size =8
lr = [0.0001, 0.001, 0.01, 0.1, 1]
mom = [0.5, 0.9, 0.99]
test_loss = []
train_loss = []
test_accuracy = []
train_accuracy = []
history_loss = []
history_val_loss =[]
history_acc = []
history_val_acc = []
gen_error_loss = []
gen_error_acc = []
drp1=[0, 0.25, 0.5, 0.75, 1]
drp2=[0, 0.25, 0.5, 0.75, 1]
drp3=[0, 0.25, 0.5, 0.75, 1]
reg = [None, regularizers.l1(0.01)]
epochs =80

x_train = np.zeros((7291, 16, 16))
x_test = np.zeros((2007, 16, 16))
dftrain = pd.read_csv("train.csv")
dftest = pd.read_csv("test.csv")
y_train = dftrain['Class']
y_train = np.asarray(y_train)
y_test = dftest['Class']
y_test = np.asarray(y_test)
x_train_1d = dftrain[dftrain.columns.difference(['Class'])]
x_train_1d = np.asarray(x_train_1d)
x_test_1d = dftest[dftest.columns.difference(['Class'])]
x_test_1d = np.asarray(x_test_1d)
#new_arr = ((ft_np - (-1)) * (1/(arr.max() - arr.min()) * 255).astype('uint8')

for i in range(0,x_train_1d.shape[0]):
    
    x_train_1d[i] = ((x_train_1d[i] - (-1))/float(2.0)*255).astype('uint8')
    x_train[i] = np.reshape(x_train_1d[i],(16, 16))
    
for i in range(0,x_test_1d.shape[0]):
    
    x_test_1d[i] = ((x_test_1d[i] - (-1))/float(2.0)*255).astype('uint8')
    x_test[i] = np.reshape(x_test_1d[i],(16, 16))
    
w_xtrain = np.zeros((x_train.shape))
new_xtrain = np.zeros((x_train.shape[0], 256))
    
M = np.float32([[1,0,2],[0,1,0]])
for i in range(0,x_train.shape[0]):

    w_xtrain[i] = cv2.warpAffine(x_train[i], M,(img_rows, img_cols))
    new_xtrain[i] = w_xtrain[i].flatten()
    new_xtrain[i] = new_xtrain[i]/255
    new_xtrain[i] = 2*(new_xtrain[i])-1
    
x_trai = np.zeros((x_train.shape[0],256))
for i in range(0,x_train.shape[0]):

    x_trai[i] = x_train[i].flatten()
    x_trai[i] = x_trai[i]/255
    x_trai[i] = 2*(x_trai[i])-1
    
x_trai_labels = np.zeros((x_trai.shape[0],1))    
x_trai_labels = y_train
x_trai_labels = x_trai_labels.reshape(x_trai_labels.shape[0], 1)

reverted_xtrain = np.concatenate((x_trai, x_trai_labels), axis = 1)
modified_xtrain = np.concatenate((new_xtrain, x_trai_labels), axis =1)
np.savetxt("reverted_xtrain.txt", reverted_xtrain)
np.savetxt("modified_xtrain.txt",modified_xtrain)
final_new_xtrain = np.concatenate((reverted_xtrain, modified_xtrain), axis = 0)
np.savetxt("final_new_xtrain.txt", final_new_xtrain)



x_t = np.zeros((x_test.shape[0],256))
for i in range(0,x_test.shape[0]):

    x_t[i] = x_test[i].flatten()
    x_t[i] = x_t[i]/255
    x_t[i] = 2*(x_t[i])-1
    
x_trai_labels = y_test
x_trai_labels = x_trai_labels.reshape(x_trai_labels.shape[0], 1)   
reverted_test = np.concatenate((x_t, x_trai_labels), axis = 1)
np.savetxt("reverted_test.txt", reverted_test)