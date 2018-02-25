import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import load_model

tardir = '../../wikifacemini/'
y = np.load(tardir+"wikifacemini.npy")
x = np.load("gaussianblur.npy")

model = load_model('model.h5')

model.summary()
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.0001))
model.fit(x,y,batch_size=128,epochs=30)

model.save('model.h5')