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

num_classes = 10
#batch_size = 10
epochs = 100

data = np.loadtxt("zip_train.txt")
index = 0
train_label = data[:,index]
train_dat = np.delete(data, index, axis=1)
train_data = (train_dat-(-1))/2

data1 = np.loadtxt("zip_test.txt")
test_label = data1[:,index]
test_dat = np.delete(data1, index, axis=1)
test_data = (test_dat-(-1))/2

## convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

batch_size =[ 2048, 256, 128, 64]
lr = [0.0001, 0.001, 0.01, 0.1, 0.9,1, 2, 10]
#kernels = ["glorot_uniform", "random_uniform", "zeros", "uniform", "lecun_normal", "he_uniform" ]
#biases =["zeros","ones","random_uniform" ]
#kernels = ["glorot_uniform"]
#biases =["zeros"]

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

def dense_model(batch_size):
    model = Sequential()
    opti = optimizers.SGD(lr=0.9)
    model.add(Dense(256,
                    input_shape=(256,),
                    activation='relu',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "random_uniform"
                    ))
    model.add(Dense(256,
                    activation='relu',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "zeros"
                    ))
    model.add(Dense(256,
                    activation='sigmoid',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "random_uniform"
                    ))
    model.add(Dense(num_classes,
                    activation='softmax',
                    kernel_initializer = "he_uniform", 
                    bias_initializer = "ones"
                    ))
    
    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    model.summary()
    
    history=model.fit(train_data,train_label,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_data=(test_data, test_label)
                      )
    score1 = model.evaluate(test_data, test_label, verbose=0)
#    print('Test loss:', score1[0])
#    print('Test accuracy:', score1[1])
    test_loss.append(score1[0])
    test_accuracy.append(score1[1])
    
    
    
    score2 = model.evaluate(train_data, train_label, verbose=0)
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
    
for i in range(0,len(batch_size)):
    dense_model(batch_size[i])


fig = plt.figure(dpi=200)
plt.title("Loss vs Epoch for Train and Test")
plt.xlabel("Epochs")
plt.ylabel(" Loss")
color=iter(cm.rainbow(np.linspace(0,1,2*len(batch_size))))
for i in range(0,len(batch_size)):
    c = next(color)
    plt.plot(history_loss[i], label = "Training Loss_"+str(batch_size[i]), color = c)
    c = next(color)
    plt.plot(history_val_loss[i], label = "Test Loss_"+str(batch_size[i]), color = c)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
plt.grid(True)
plt.savefig('Fig_Loss_Dense_Batch_Size.jpg') 
plt.show()

#plt.figure()
fig = plt.figure(dpi=200)#
plt.title("Accuracy vs Epoch for Train and Test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
color=iter(cm.rainbow(np.linspace(0,1,2*len(batch_size))))
for i in range(0, len(batch_size)):
    c = next(color)
    plt.plot(history_acc[i], label = "Training Accuracy_"+str(batch_size[i]), color = c)
    c = next(color)
    plt.plot(history_val_acc[i], label = "Test Accuracy_"+str(batch_size[i]), color = c)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
plt.grid(True)
plt.savefig('Fig_Acc_Dense_Batch_Size.jpg')  
    

plt.show()


replicate = "Dense_Batch_Size"
with open("output_data_" + str(replicate) + ".csv", "w") as out_file:
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
        
            
