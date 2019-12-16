# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:56:25 2018

@author: altri
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib.pyplot import cm 
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd

num_classes = 10
#batch_size = 10
epochs = 50

x_train = np.zeros((7291, 16, 16))
x_test = np.zeros((2007, 16, 16))
dftrain = pd.read_csv("../final_modified_train.csv")
dftrain2 = pd.read_csv("../reverted_train.csv")
dftest = pd.read_csv("../reverted_test.csv")
y_train = dftrain['Class']
y_train2 = dftrain2['Class']

y_test = dftest['Class']

train_data = dftrain[dftrain.columns.difference(['Class'])]
train_data2 = dftrain2[dftrain.columns.difference(['Class'])]
#x_train_1d = np.asarray(x_train_1d)
test_data = dftest[dftest.columns.difference(['Class'])]

## convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(y_train, num_classes)
train_label2 = keras.utils.to_categorical(y_train2, num_classes)

test_label = keras.utils.to_categorical(y_test, num_classes)

#batch_size =[ 4096, 2048, 1024, 512, 256, 64, 32, 16, 8, 4, 2, 1]
#lr = [0.0001, 0.001, 0.01, 0.1, 0.9,1, 2, 10]
#kernels = ["glorot_uniform", "random_uniform", "zeros", "uniform", "lecun_normal", "he_uniform" ]
#biases =["zeros","ones","random_uniform" ]
#kernels = ["glorot_uniform"]
#biases =["zeros"]
momentum = [0.5, 0.9, 0.99]

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

def dense_model(data, label):
    model = Sequential()
    opti = optimizers.SGD(lr=0.05, momentum = 0.9)
    model.add(Dense(256,
                    input_shape=(256,),
                    activation='sigmoid',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "random_uniform"
                    ))
    model.add(Dense(256,
                    activation='relu',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "ones"
                    ))
    model.add(Dense(256,
                    activation='sigmoid',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "random_uniform"
                    ))
    model.add(Dense(num_classes,
                    activation='softmax',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "zeros"
                    ))
    
    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    model.summary()
    
    history=model.fit(data,label,
                      batch_size=64,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(test_data, test_label)
                      )
    score1 = model.evaluate(test_data, test_label, verbose=0)
#    print('Test loss:', score1[0])
#    print('Test accuracy:', score1[1])
    test_loss.append(score1[0])
    test_accuracy.append(score1[1])
    
    
    
    score2 = model.evaluate(data,label, verbose=0)
#    print('Train loss:', score2[0])
#    print('Train accuracy:', score2[1])
    train_loss.append(score2[0])
    train_accuracy.append(score2[1])
    history_loss.append(history.history['loss'])
    history_val_loss.append(history.history['val_loss'])
    history_acc.append(history.history['acc'])
    history_val_acc.append(history.history['val_acc'])


#for i in range(0, len(kernels)):
#    for j in range(0, len(biases)):
#        dense_model(kernels[i], biases[j])
    

dense_model(train_data, train_label)
dense_model(train_data2, train_label2)


#dense_model()

fig = plt.figure(dpi=200)
plt.title("Loss vs Epoch for Train and Test Adversarial")
plt.xlabel("Epochs")
plt.ylabel(" Loss")
color=iter(cm.rainbow(np.linspace(0,1,2*len(momentum))))
for i in range(0,1):
    c = next(color)
    plt.plot(history_loss[0], label = "Test Loss_ Adversarial", color = c)
    c = next(color)
    plt.plot(history_val_loss[1], label = "Test Loss", color = c)
    
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
plt.grid(True)
plt.savefig('Fig_Loss_momentum.jpg') 
plt.show()

#plt.figure()
fig = plt.figure(dpi=200)#
plt.title("Accuracy vs Epoch for Train and Test ")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
color=iter(cm.rainbow(np.linspace(0,1,4)))
for i in range(0, 1):
    c = next(color)
    plt.plot(history_val_acc[0], label = "Test Accuracy_Adversarial", color = c)
    c = next(color)
    plt.plot(history_val_acc[1], label = "Test Accuracy", color = c)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
plt.grid(True)
plt.savefig('Fig_Acc_momentum.jpg')  
    

plt.show()


replicate = "momemtum"
with open("output_data_dense" + str(replicate) + ".csv", "w") as out_file:
    for i in range(len(test_loss)):
        out_str = ""
        out_str += str(train_loss[i])
        out_str += "," + str(test_loss[i])
        out_str += "," + str(train_accuracy[i])
        out_str += "," + str(test_accuracy[i])
        out_str += "," + str(train_loss[i]-test_loss[i])
        out_str += "," + str(train_accuracy[i]-test_accuracy[i])
        out_str += "\n"
        out_file.write(out_str)
        
            
