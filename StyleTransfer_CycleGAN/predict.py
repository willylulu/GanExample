import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import io, transform
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from util import InstanceNormalization2D

x2y = load_model('models/generator_x2y21.h5', custom_objects={'InstanceNormalization2D':InstanceNormalization2D})
y2x = load_model('models/generator_y2x21.h5', custom_objects={'InstanceNormalization2D':InstanceNormalization2D})

pathX = sys.argv[1]
pathY = sys.argv[2]
testX = os.listdir(pathX)
testY = os.listdir(pathY)

seedX = np.random.permutation(testX)
seedY = np.random.permutation(testY)

picX = [ io.imread(os.path.join(pathX, x)) for x in seedX[:16]]
picY = [ io.imread(os.path.join(pathY, y)) for y in seedY[:16]]

arrX = np.array(picX)/127.5-1
arrY = np.array(picY)/127.5-1

arrX2Y = x2y.predict(arrX)
arrY2X = y2x.predict(arrY)

images = np.concatenate([arrX[:8], arrX2Y[:8], arrY[:8], arrY2X[:8], arrX[8:], arrX2Y[8:], arrY[8:], arrY2X[8:]])
width = 8
height = 4
new_im = Image.new('RGB', (256*height,256*width))
for ii in range(height):
    for jj in range(width):
        index=ii*width+jj
        image = (images[index]/2+0.5)*255
        image = image.astype(np.uint8)
        new_im.paste(Image.fromarray(image,"RGB"), (256*ii,256*jj))
filename = "test.jpg"
new_im.save(filename)