# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:14:41 2019

@author: Fahad

Libraries needed for this pogram
    tensorflow 2.0.0
    numpy 1.17.2 
    
"""
########################   Import Libaries - Start ############################################################## 


from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import time
#import matplotlib.pyplot as plt

########################   Import data ############################################################## 

import load_data
data = load_data.read_data_sets(one_hot=True)
train = data.train.data
test  = data.test.data
train_labels = data.train.labels
test_labels = data.test.labels

#datatotal = np.append(train,test,axis=1)

########################   AutoEncoder - Start ############################################################## 

start = time.process_time()
 
input_data = Input(shape=(310,))
encoded = Dense(205, activation='relu')(input_data)

encoded = Dense(100, activation='relu')(encoded)

decoded = Dense(205, activation='relu')(encoded)
decoded = Dense(310, activation='relu')(decoded)

autoencoder = Model(input_data, decoded)
encoder = Model(input_data, encoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
autoencoder.fit(train, train, epochs=50, batch_size=1024, shuffle=True, validation_data=(test,test))

encoder_time = time.process_time() - start

########################   Classifier - Start ############################################################## 

start = time.process_time()

encoded_test = encoder.predict(test)
encoded_train = encoder.predict(train)
#np.savetxt("encoded_test.csv", encoded_test, delimiter=",")
#np.savetxt("encoded_train.csv", encoded_train, delimiter=",")
#encoded_test = np.genfromtxt('encoded_test.csv', delimiter=',')
#encoded_train = np.genfromtxt('encoded_train.csv', delimiter=',')
input_data = Input(shape=(100,))
classify = BatchNormalization()(input_data)
classify = Dense(70, activation='relu')(classify)
classify = Dense(40, activation='relu')(classify)
classify = Dense(10, activation='relu')(classify)
#classify = Dropout(0.5)(classify)
classify = Dense(3, activation='softmax')(classify)
classifier = Model(input_data, classify)
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
classifier.fit(encoded_train, train_labels, epochs=15, batch_size=1024, shuffle=True, validation_split=0.05)
#classifier.fit(train, train_labels, epochs=15, batch_size=10, shuffle=True, validation_split=0.05)


########################   Classification - Accuracy ######################################################## 


print("time taken by encoder : ", encoder_time)
print("time taken by Classifier : ",time.process_time() - start)

final_test = classifier.predict(encoded_test)
count = 0
for i in range(0, len(test_labels)):
    if(np.where(final_test[i] == np.max(final_test[i])) == np.where(test_labels[i] == np.max(test_labels[i]))):
        count+=1
#    else:
#        print(i)
print("Classification accuracy on Testing data : ",count/len(test_labels)*100,"%")    

final_test = classifier.predict(encoded_train)
count = 0
for i in range(0, len(train_labels)):
    if(np.where(final_test[i] == np.max(final_test[i]))[0][0] == np.where(train_labels[i] == np.max(train_labels[i]))[0][0]):
        count+=1
#    else:
#        print(i)
print("Classification accuracy on training data : ", count/len(train_labels)*100,"%")

########################   pogram - end ############################################################## 