import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import io, transform
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from util import InstanceNormalization2D

x2y = load_model('models/generator_x2y18.h5', custom_objects={'InstanceNormalization2D':InstanceNormalization2D})
y2x = load_model('models/generator_y2x18.h5', custom_objects={'InstanceNormalization2D':InstanceNormalization2D})

pathY = sys.argv[1]

picY = transform.resize(io.imread(pathY),(256,256))

arrY = picY*2-1
# arrY = np.array(picY)*2-1
arrY = np.expand_dims(arrY, 0)

arrY2X = y2x.predict(arrY)

image = arrY2X/2+0.5

filename = "test.jpg"
io.imsave(filename, image[0])