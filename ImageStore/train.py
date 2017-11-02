import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Conv2DTranspose, Activation
from keras.layers import Conv2D, MaxPooling2D

tardir = '../../wikifacemini/'
y = np.load(tardir+"wikifacemini.npy")
x = np.load(tardir+"gaussianblur.npy")

model = Sequential()

model.add(Conv2D(64,(3,3),strides=(2,2),padding='same',input_shape=(64,64,3)))
model.add(LeakyReLU())
model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
model.add(LeakyReLU())
model.add(Conv2D(256,(3,3),strides=(2,2),padding='same'))
model.add(LeakyReLU())
model.add(Conv2D(512,(3,3),strides=(2,2),padding='same'))
model.add(LeakyReLU())
model.add(Activation('sigmoid'))


model.add(Conv2DTranspose(256,(3,3),strides=(2,2),padding='same',output_shape=(8,8,256),activation='relu'))
model.add(Conv2DTranspose(128,(3,3),strides=(2,2),padding='same',output_shape=(16,16,128),activation='relu'))
model.add(Conv2DTranspose(64,(3,3),strides=(2,2),padding='same',output_shape=(32,32,64),activation='relu'))
model.add(Conv2DTranspose(3,(3,3),strides=(2,2),padding='same',output_shape=(64,64,3),activation='relu'))
model.add(Activation('tanh'))

model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x,y,batch_size=128,epochs=30)

model.save('model.h5')