import sys
import numpy as np
from CycleGan import cycleGan

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

K.set_learning_phase(False)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print("load domain X...")
monet = np.load(sys.argv[1])
print("load domain Y...")
photo = np.load(sys.argv[2])
print("finish")

cyclegan = cycleGan()
cyclegan.train(monet, photo)