import sys
import numpy as np
from StackGan_v2 import StackGan

import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

K.set_learning_phase(True)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

stackgan = StackGan(sys.argv[1])
stackgan.train()