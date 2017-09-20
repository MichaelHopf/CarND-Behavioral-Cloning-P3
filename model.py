
# coding: utf-8

# In[ ]:


import csv
#import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
get_ipython().magic('matplotlib inline')


################################################################################################################
# Import Data
################################################################################################################
images = []
measurements = []
lines0 = []
S = np.array([0, 1, -1])

for i in range(8):
    print(i+1)
    lines = []
    with open('./data{}/driving_log.csv'.format(i+1)) as csvfile: # The data is organized in different folders
        reader = csv.reader(csvfile)
        for line in reader:            
            lines.append(line)
            for k in range(3):
                lines0.append(line)
 
print("{} pictures, i.e., {} for training and {} for validation.".format(3*len(lines0),round(0.8*3*len(lines0)),round(0.2*3*len(lines0))))
print("With flipping, we have {}  images for training.".format(round(0.8*3*2*len(lines0))))
print("If we only use the center image and flipping, we have {} images for training.".format(round(0.8*2*len(lines0))))


################################################################################################################
# Splitting data in train and validation set
################################################################################################################
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines0, test_size=0.2)


################################################################################################################
# Data generator. Data augmentation is done here.
################################################################################################################
import sklearn
import os
from random import shuffle

# This function is used to identify the correct folder. Prepare for some ugly code.
def tryout(name):
    try:
        cur_path = './data1/IMG/' + name
        img = mpimg.imread(cur_path)
    except:
        try:
            cur_path = './data2/IMG/' + name
            img = mpimg.imread(cur_path)
        except:
            try:
                cur_path = './data3/IMG/' + name
                img = mpimg.imread(cur_path)
            except:
                try:
                    cur_path = './data4/IMG/' + name
                    img = mpimg.imread(cur_path)
                except:
                    try:
                        cur_path = './data5/IMG/' + name
                        img = mpimg.imread(cur_path)
                    except:
                        try:
                            cur_path = './data6/IMG/' + name
                            img = mpimg.imread(cur_path)
                        except:
                            try:
                                cur_path = './data7/IMG/' + name
                                img = mpimg.imread(cur_path)
                            except:
                                cur_path = './data8/IMG/' + name
                                img = mpimg.imread(cur_path)
    return img

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                for k in range(3):
                    name = batch_sample[k].split('\\')[-1]
                    center_image = tryout(name)
                    center_image_flipped = np.fliplr(center_image)                                  
                    center_angle = float(batch_sample[3]) + S[k] * 0.05 # steering angle correction
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(center_image_flipped)
                    angles.append(-center_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


################################################################################################################
# Defining the train and validation generators
################################################################################################################
train_generator = generator(train_samples, batch_size = 8)
validation_generator = generator(validation_samples, batch_size = 8)
factor = 6 # factor by which the data increased in the generator routine


################################################################################################################
# Building and Training the model
################################################################################################################
from keras.models import Sequential
from keras import *
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Lambda, Flatten, Dense, Reshape, Input, merge, normalization, Cropping2D
from keras.models import Model

inp = Input(shape=(160,320,3))
layer_0  = Cropping2D(cropping=((80,20), (0,0)))(inp)
layer_1  = Lambda(lambda x: x/127.5 - 1)(layer_0)
layer_2  = Lambda(lambda x: -x)(layer_1)
layer_3  = MaxPooling2D()(layer_2)
layer_4  = MaxPooling2D((1,4))(layer_3)
layer_5  = Lambda(lambda x: -x)(layer_4)
conv1    = Conv2D(30,3,3, border_mode='same', activation='relu')(layer_4) #100
conv2    = Conv2D(15,1,1, border_mode='same', activation='relu')(layer_4) #50
conv3    = Conv2D(15,5,5, border_mode='same', activation='relu')(layer_4) #50
merge1   = merge([conv1, conv2, conv3], mode = 'concat', concat_axis = -1)
layer_6  = MaxPooling2D()(merge1)
layer_6  = normalization.BatchNormalization()(layer_6)
layer_8  = Conv2D(20,3,3, border_mode='same', activation='relu')(layer_6) #20
layer_10 = MaxPooling2D()(layer_8)
layer_10 = normalization.BatchNormalization()(layer_10)
layer_10b= Conv2D(10,3,3, border_mode='same', activation='relu')(layer_10) #20
layer_11 = Flatten()(layer_10b)
layer_12 = Dense(50, activation='relu')(layer_11)
layer_13 = Dense(30, activation='relu')(layer_12)
outlayer = Dense(1)(layer_13)

model = model = Model(input=inp, output=outlayer)
model.summary()

model.compile(loss = 'mse', optimizer='adam')
checkpointer = callbacks.ModelCheckpoint(filepath='./model_both_17.hdf5', verbose=1, save_best_only=True)
history_object = model.fit_generator(train_generator, samples_per_epoch = factor*len(train_samples), validation_data = validation_generator, nb_val_samples = factor*len(validation_samples), nb_epoch = 10, callbacks=[checkpointer])

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

