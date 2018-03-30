import tensorflow as tf
import keras
from keras.layers import LeakyReLU, Conv2D, Add, ZeroPadding2D, Activation
from util import InstanceNormalization2D

truncate_normal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)

def residual_block(x, dim, ks):
    y = Conv2D(dim, ks, strides=1, padding="same", kernel_initializer=truncate_normal)(x)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)
    y = Conv2D(dim, ks, strides=1, padding="same", kernel_initializer=truncate_normal)(y)
    y = InstanceNormalization2D()(y)
    return Add()([x,y])
    