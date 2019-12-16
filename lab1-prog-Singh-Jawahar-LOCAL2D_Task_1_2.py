# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:12:03 2018

@author: PhObiA
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:40:05 2018

@author: altri
"""

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import MaxPooling2D, LocallyConnected2D
from keras import backend as K
import pandas as pd
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 


img_rows = 16
img_cols = 16
num_classes = 10
batch_size =[ 4096, 2048, 1024, 512, 256, 64, 32, 16, 8, 4, 2, 1]
lr = [0.0001, 0.001, 0.01, 0.1, 0.9,1, 2, 10]
#kernels = ["glorot_uniform", "random_uniform", "zeros", "lecun_normal", "he_uniform" ]
#kernels = ["he_uniform"]
#biases1 =["zeros","ones","random_uniform" ]
#biases2 =["zeros","ones","random_uniform" ]
#biases3 =["zeros","ones","random_uniform" ]
#biases4 =["zeros","ones","random_uniform" ]

biases1 = ["relu", "sigmoid"]

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

epochs =50

x_train = np.zeros((7291, 16, 16))
x_test = np.zeros((2007, 16, 16))
dftrain = pd.read_csv("train.csv")
dftest = pd.read_csv("test.csv")
y_train = dftrain['Class']
y_test = dftest['Class']
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
   
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def local_model(a1, a2, a3, a4):
    model = Sequential()
    adel = optimizers.SGD(lr = 1.0)
    model.add(LocallyConnected2D(64, (3, 3), input_shape=input_shape, activation = a1))
    model.add(LocallyConnected2D(32, (3, 3), activation=a2))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LocallyConnected2D(32, (3, 3), activation=a3))
    model.add(Flatten())
    model.add(Dense(num_classes, activation=a4))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adel,
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
              batch_size=128,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))
    
    
    score1 = model.evaluate(x_test, y_test, verbose=0)
#    print('Test loss:', score1[0])
#    print('Test accuracy:', score1[1])
    test_loss.append(score1[0])
    test_accuracy.append(score1[1])
    
    score2 = model.evaluate(x_train, y_train, verbose=0)
#    print('Train loss:', score2[0])
#    print('Train accuracy:', score2[1])
    train_loss.append(score2[0])
    train_accuracy.append(score2[1])
    history_loss.append(history.history['loss'])
    history_val_loss.append(history.history['val_loss'])
    history_acc.append(history.history['acc'])
    history_val_acc.append(history.history['val_acc'])
    
    print(model.summary())
 
    

for j in range(0, len(biases1)):
    for k in range(0, len(biases1)):
        for l in range(0, len(biases1)):
            for m in range(0, len(biases1)):
                local_model(biases1[j], biases1[k], biases1[l], biases1[m])
                
replicate = "activations"
with open("output_locally_" + str(replicate) + ".csv", "w") as out_file:
    for i in range(len(test_loss)):
        
        out_str = ""
        out_str += str(train_loss[i])
        out_str += "," + str(test_loss[i])
        out_str += "," + str(train_accuracy[i])
        out_str += "," + str(test_accuracy[i])
        out_str += "\n"
        out_file.write(out_str)
with open("output_locally_names" + str(replicate) + ".csv", "w") as out_fle:

    for j in range(0, len(biases1)):
        for k in range(0, len(biases1)):
            for l in range(0, len(biases1)):
                for m in range(0, len(biases1)):
                    out_str += str(biases1[j])
                    out_str += str(biases1[k])
                    out_str += str(biases1[l])
                    out_str += str(biases1[m])
    
    out_fle.write(out_str)          

        
fig = plt.figure(dpi=200)
plt.title("Loss vs Epoch for Train and Test without Random Labels")
plt.xlabel("Epochs")
plt.ylabel(" Loss")
color=iter(cm.rainbow(np.linspace(0,2,2*len(biases1)*len(kernels))))
for i in range(0, len(kernels)):
    for j in range(0, len(biases1)):
        for k in range(0, len(biases2)):
            for l in range(0, len(biases3)):
                for m in range(0, len(biases4)):
                    #c = next(color)
                    #plt.plot(history_loss[i], label = "Training Loss_lr"+str(kernels[i])+""+str(biases[j]), color = c)
                    c = next(color)
                    plt.plot(history_val_loss[i], label = "Test Loss_"+str(kernels[i])+"_"+str(biases1[j])+"_"+str(biases2[k])+"_"+str(biases3[l])+"_"+str(biases4[m]), color = c)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
          fancybox=True, shadow=True)
plt.grid(True)
plt.savefig('Fig_Loss_Local_KB.jpg') 
plt.show()

#plt.figure()
fig = plt.figure(dpi=200)#
plt.title("Accuracy vs Epoch for Train and Test without Random Labels")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
color=iter(cm.rainbow(np.linspace(0,1,2*len(biases1)*len(kernels))))
for i in range(0, len(kernels)):
    for j in range(0, len(biases1)):
        for k in range(0, len(biases2)):
            for l in range(0, len(biases3)):
                for m in range(0, len(biases4)):
                    #c = next(color)
                    #plt.plot(history_acc[i], label = "Training Accuracy_lr"+str(kernels[i])+""+str(biases[j]), color = c)
                    c = next(color)
                    plt.plot(history_val_loss[i], label = "Test Accuracy_"+str(kernels[i])+"_"+str(biases1[j])+"_"+str(biases2[k])+"_"+str(biases3[l])+"_"+str(biases4[m]), color = c)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2,
          fancybox=True, shadow=True)
plt.grid(True)
plt.savefig('Fig_Acc_Local_KB.jpg')  
    
plt.show()
        
replicate = "Kernel_Bias"
with open("output_data_param_" + str(replicate) + ".csv", "w") as out_file:
    for i in range(len(test_loss)):
        
        out_str = ""
        out_str += str(train_loss[i])
        out_str += "," + str(test_loss[i])
        out_str += "," + str(train_accuracy[i])
        out_str += "," + str(test_accuracy[i])
        out_str += "\n"
        out_file.write(out_str)