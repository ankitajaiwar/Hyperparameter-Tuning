# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 00:06:51 2018

@author: Ankita
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:28:55 2018

@author: Ankita
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:51:40 2018

@author: Ankita
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 17:42:55 2018

@author: Ankita
"""

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

import cv2

img_rows = 16
img_cols = 16
num_classes = 10
batch_size =[ 4096, 2048, 1024, 512, 256, 64, 32, 16, 8, 4, 2, 1]
lr = [0.0001, 0.001, 0.01, 0.1, 0.9,1, 2, 10]
kernels = ["glorot_uniform", "random_uniform", "zeros", "lecun_normal",  ]

biases =["zeros","ones","random_uniform" ]
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

epochs =80

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


def cnn_model(kernel_initializer1, bias_initializer1,kernel_initializer2, bias_initializer2,kernel_initializer3, bias_initializer3,kernel_initializer4, bias_initializer4 ):
    

    model = Sequential()
    adel = optimizers.SGD()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,  kernel_initializer = kernel_initializer1, bias_initializer = bias_initializer1))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer = kernel_initializer2, bias_initializer = bias_initializer2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer = kernel_initializer3, bias_initializer = bias_initializer3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid', kernel_initializer = kernel_initializer4, bias_initializer = bias_initializer4))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
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

       
cnn_model("zeros", "ones","he_uniform", "random_uniform","he_uniform", "zeros","glorot_uniform", "zeros")
cnn_model("he_uniform", "zeros", "random_uniform", "zeros", "zeros", "ones", "zeros", "zeros")
cnn_model("lecun_normal", "zeros", "glorot_uniform", "ones", "glorot_uniform", "ones", "glorot_uniform", "zeros")
        
        
replicate = "parameters_kernels_biases"
with open("output_data_" + str(replicate) + ".csv", "w") as out_file:
    for i in range(len(test_loss)):
        
        out_str = ""
        out_str += str(train_loss[i])
        out_str += "," + str(test_loss[i])
        out_str += "," + str(train_accuracy[i])
        out_str += "," + str(test_accuracy[i])
        out_str += "\n"
        out_file.write(out_str)
78
with open("output_data_param_names" + str(replicate) + ".csv", "w") as out_fil:        
       
    
    for i in range(0, len(kernels)):
        for j in range(0, len(biases)):
                
            for k in range(0, len(kernels)):
                for l in range(0, len(biases)):
                
                    for m in range(0, len(kernels)):
                        for n in range(0, len(biases)):
                
                            for o in range(0, len(kernels)):
                                for p in range(0, len(biases)):
                                    out_str += "," + str(kernels[i])
                                    out_str += "," + str(biases[j])
                                    out_str += "," + str(kernels[k])
                                    out_str += "," + str(biases[l])
                                    out_str += "," + str(kernels[m])
                                    out_str += "," + str(biases[n])
                                    out_str += "," + str(kernels[o])
                                    out_str += "," + str(biases[p])
                                    
                                    out_str += "\n"
                            
    out_fil.write(out_str)

fig = plt.figure(dpi =150)
plt.title("Loss vs Epoch for Test")
plt.xlabel("Epochs")
plt.ylabel(" Loss")
color=iter(cm.rainbow(np.linspace(0,2,len(biases)*3)))


c = next(color)
   
plt.plot(history_val_loss[0], label ="Model1", color = c)
c = next(color)
plt.plot(history_val_loss[1], label = "Model2", color = c)
c = next(color)
plt.plot(history_val_loss[2], label = "Model3", color = c)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
plt.grid(True)

plt.show()


fig = plt.figure(dpi =150)
plt.title("Accuracy vs Epoch for Test")
plt.xlabel("Epochs")
plt.ylabel(" Loss")
color=iter(cm.rainbow(np.linspace(0,2,len(biases)*3)))


c = next(color)
   
plt.plot(history_val_acc[0], label ="Model1", color = c)
c = next(color)
plt.plot(history_val_acc[1], label = "Model2", color = c)
c = next(color)
plt.plot(history_val_acc[2], label = "Model3", color = c)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)
plt.grid(True)

plt.show()
