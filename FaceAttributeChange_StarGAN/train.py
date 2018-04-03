import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from starGan import StarGan

# K.set_floatx('float64')
K.set_learning_phase(False)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

starGan = StarGan(sys.argv[1])
starGan.train()