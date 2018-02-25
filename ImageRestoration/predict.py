import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import load_model

#tardir = '../../wikifacemini/'
x = np.load(sys.argv[1])
y = np.load(sys.argv[2])

model = load_model('model.h5')
model.summary()

result = model.predict(x,batch_size=128,verbose=0)

f,ax = plt.subplots(8,9)
for i in range(8):
    for j in range(3):
        index=i*8+j
        ax[i,j*3].imshow(result[index])
        ax[i,j*3+1].imshow(y[index])
        ax[i,j*3+2].imshow(x[index])
plt.show()