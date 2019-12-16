# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:22:57 2018

@author: altri
"""
# R - Relu
# Model 1 - S R S
# Model 2 - S R R
# Model 3 - S S R
# Model 4 - R R S
# Model 5 - R S R
# Model 6 - R S S

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from matplotlib.pyplot import cm 
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
#test_label = keras.utils.to_categorical(test_label, num_classes)

#X_train1, X_test1, y_train1, y_test1 = train_test_split(train_data, train_label, test_size=0.25, random_state=42)
#X_train2, X_test2, y_train2, y_test2 = train_test_split(train_data, train_label, test_size=0.25, random_state=35)
#X_train3, X_test3, y_train3, y_test3 = train_test_split(train_data, train_label, test_size=0.25, random_state=25)
#X_train4, X_test4, y_train4, y_test4 = train_test_split(train_data, train_label, test_size=0.25, random_state=50)
#X_train5, X_test5, y_train5, y_test5 = train_test_split(train_data, train_label, test_size=0.25, random_state=20)
#X_train6, X_test6, y_train6, y_test6 = train_test_split(train_data, train_label, test_size=0.25, random_state=15)


out = []
predicted = []

#def dense_bagging():
    
#model 1
model1 = Sequential()
opti = optimizers.SGD(lr=0.01,momentum=0.09)
model1.add(Dense(256,
                input_shape=(256,),
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model1.add(Dense(256,
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "ones"
                ))
model1.add(Dense(256,
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model1.add(Dense(num_classes,
                activation='softmax',
                kernel_initializer = "he_uniform", 
                bias_initializer = "zeros"
                ))

model1.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
model1.summary()

history1=model1.fit(train_data,train_label,
                  batch_size=256,
                  epochs=epochs,
                  verbose=2
                  )


# model 2 (Layer 3 uses relu as activation)
model2 = Sequential()
opti = optimizers.SGD(lr=0.01,momentum=0.09)
model2.add(Dense(256,
                input_shape=(256,),
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model2.add(Dense(256,
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "ones"
                ))
model2.add(Dense(256,
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model2.add(Dense(num_classes,
                activation='softmax',
                kernel_initializer = "he_uniform", 
                bias_initializer = "zeros"
                ))

model2.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
model2.summary()

history2=model2.fit(train_data,train_label,
                  batch_size=8,
                  epochs=epochs,
                  verbose=2
                  )


# model 3 (Layer 3 uses relu and Layer 2 uses sigmoid as activation)
model3 = Sequential()
opti = optimizers.SGD(lr=0.01,momentum=0.09)
model3.add(Dense(256,
                input_shape=(256,),
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model3.add(Dense(256,
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "ones"
                ))
model3.add(Dense(256,
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model3.add(Dense(num_classes,
                activation='softmax',
                kernel_initializer = "he_uniform", 
                bias_initializer = "zeros"
                ))

model3.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
model3.summary()

history3=model3.fit(train_data,train_label,
                  batch_size=8,
                  epochs=epochs,
                  verbose=2
                  )

#model 4 (Layer 1 uses relu)
model4 = Sequential()
opti = optimizers.SGD(lr=0.01,momentum=0.09)
model4.add(Dense(256,
                input_shape=(256,),
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model4.add(Dense(256,
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "ones"
                ))
model4.add(Dense(256,
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model4.add(Dense(num_classes,
                activation='softmax',
                kernel_initializer = "he_uniform", 
                bias_initializer = "zeros"
                ))

model4.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
model4.summary()

history4=model4.fit(train_data,train_label,
                  batch_size=8,
                  epochs=epochs,
                  verbose=2
                  )

#model 5 (Layer 1 relu, layer 2 sigmoid, layer 3 relu)
model5 = Sequential()
opti = optimizers.SGD(lr=0.01,momentum=0.09)
model5.add(Dense(256,
                input_shape=(256,),
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model5.add(Dense(256,
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "ones"
                ))
model5.add(Dense(256,
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model5.add(Dense(num_classes,
                activation='softmax',
                kernel_initializer = "he_uniform", 
                bias_initializer = "zeros"
                ))

model5.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
model5.summary()

history5=model5.fit(train_data,train_label,
                  batch_size=8,
                  epochs=epochs,
                  verbose=2
                  )


#model 6 (Layer 1 relu, layer 2 sigmoid, layer 3 sigmoid)
model6 = Sequential()
opti = optimizers.SGD(lr=0.01,momentum=0.09)
model6.add(Dense(256,
                input_shape=(256,),
                activation='relu',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model6.add(Dense(256,
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "ones"
                ))
model6.add(Dense(256,
                activation='sigmoid',
                kernel_initializer = "he_uniform", 
                bias_initializer = "random_uniform"
                ))
model6.add(Dense(num_classes,
                activation='softmax',
                kernel_initializer = "he_uniform", 
                bias_initializer = "zeros"
                ))

model6.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
model6.summary()

history6=model6.fit(train_data,train_label,
                  batch_size=8,
                  epochs=epochs,
                  verbose=2
                  )


pred1 = model1.predict(test_data)
label1 = pred1.argmax(axis=-1)
pred2 = model2.predict(test_data)
label2 = pred2.argmax(axis=-1)
pred3 = model3.predict(test_data)
label3 = pred3.argmax(axis=-1)
pred4 = model4.predict(test_data)
label4 = pred4.argmax(axis=-1)
pred5 = model5.predict(test_data)
label5 = pred5.argmax(axis=-1)
pred6 = model6.predict(test_data)
label6 = pred6.argmax(axis=-1)


for i in range(0, label1.shape[0]):
    lst = [label1[i],label2[i],label3[i],label4[i],label5[i],label6[i]]
    output = max(lst, key=lst.count)
    out.append(output)
    
#    for i in range(0,test_data.shape[0]):
#        print(i)
#        print(test_data[i].shape)
#        pred1 = model1.predict(test_data[i])
#        label1 = pred1.argmax(axis=-1)
#        pred2 = model2.predict(test_data[i])
#        label2 = pred2.argmax(axis=-1)
#        pred3 = model3.predict(test_data[i])
#        label3 = pred3.argmax(axis=-1)
#        lst = [label1, label2, label3]
#        output = max(lst, key=lst.count)
#        out.append(output)
    
    
#dense_bagging()


for i in range(0,len(out)):
    if out[i] == test_label[i]:
        predicted.append(1)
    else:
        predicted.append(0)

print(predicted.count(1))
print(predicted.count(1)/len(predicted))



predicted1 = []
predicted2 = []
predicted3 = []
predicted4 = []
predicted5 = []
predicted6 = []
for i in range(0,label1.shape[0]):
    if label1[i] == test_label[i]:
        predicted1.append(1)
    else:
        predicted1.append(0)
    if label2[i] == test_label[i]:
        predicted2.append(1)
    else:
        predicted2.append(0)
    if label3[i] == test_label[i]:
        predicted3.append(1)
    else:
        predicted3.append(0)
    if label4[i] == test_label[i]:
        predicted4.append(1)
    else:
        predicted4.append(0)
    if label5[i] == test_label[i]:
        predicted5.append(1)
    else:
        predicted5.append(0)
    if label6[i] == test_label[i]:
        predicted6.append(1)
    else:
        predicted6.append(0)
        
        
print(predicted1.count(1))
print(predicted1.count(1)/len(predicted))
print(predicted2.count(1))
print(predicted2.count(1)/len(predicted))
print(predicted3.count(1))
print(predicted3.count(1)/len(predicted))
print(predicted4.count(1))
print(predicted4.count(1)/len(predicted))
print(predicted5.count(1))
print(predicted5.count(1)/len(predicted))
print(predicted6.count(1))
print(predicted6.count(1)/len(predicted))