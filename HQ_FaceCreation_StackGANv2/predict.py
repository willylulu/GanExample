import sys
import numpy as np
from StackGan_v2 import StackGan
from utils import *
import tensorflow as tf
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="0"
stackgan = StackGan(sys.argv[1], training=False)

noise = np.random.normal(0, 1.0, size=[16, 100])
fake_64, fake_128, fake_256 = stackgan.predict(noise)
saveImages('./', fake_64, 64, 0)
saveImages('./', fake_128, 128, 0)
saveImages('./', fake_256, 256, 0)