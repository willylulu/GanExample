import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation, Input, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Concatenate, Add
from keras.layers import BatchNormalization
from ops import *

d_dim = 64
g_dim = 1024
truncateNormal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)


def dnet_64(y):
    
    y = dnet_basic(y, d_dim) # 4 4 512
    y = Conv2D(1, 4, strides=4, kernel_initializer=truncateNormal)(y)

#     y = Flatten()(y)
#     y = Dense(1)(y)
    return y

def dnet_128(y):
    
    y = dnet_basic(y, d_dim) # 8 8 512
    y = dnet_conv(y, d_dim, d_dim*16)
    y = dnet_block3x3(y, d_dim, d_dim*8)
    y = Conv2D(1, 4, strides=4, kernel_initializer=truncateNormal)(y)
    
#     y = Flatten()(y)
#     y = Dense(1)(y)
    return y
    
def dnet_256(y):
    
    y = dnet_basic(y, d_dim) # 16 16 512
    y = dnet_conv(y, d_dim, d_dim*16)
    y = dnet_conv(y, d_dim, d_dim*32)
    y = dnet_block3x3(y, d_dim, d_dim*16)
    y = dnet_block3x3(y, d_dim, d_dim*8)
    y = Conv2D(1, 4, strides=4, kernel_initializer=truncateNormal)(y)
    
#     y = Flatten()(y)
#     y = Dense(1)(y)
    return y

def gnet_64(y):
    
    y = Dense(g_dim*4*4*2)(y)
    y = BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    y = Lambda(glu)(y)
    y = Reshape([4, 4, g_dim])(y) #4, 4, 1024
    
    y = gnet_upsample(y, g_dim)
    y = gnet_upsample(y, g_dim//2)
    y = gnet_upsample(y, g_dim//4)
    y = gnet_upsample(y, g_dim//8)
    
    return y

def gnet_128(y):
    y = residual_block(y, g_dim//16)
    y = residual_block(y, g_dim//16)
    y = gnet_upsample(y, g_dim//16)
    return y

def gnet_256(y):
    y = residual_block(y, g_dim//32)
    y = residual_block(y, g_dim//32)
    y = gnet_upsample(y, g_dim//32)
    return y

def gnet_64_output(y):
    
    y = Conv2D(3, 1, strides=1, kernel_initializer=truncateNormal, padding='same')(y)
    y = Activation('tanh')(y)
    return y

def gnet_128_output(y):
    
    y = Conv2D(3, 1, strides=1, kernel_initializer=truncateNormal, padding='same')(y)
    y = Activation('tanh')(y)
    return y

def gnet_256_output(y):
    
    y = Conv2D(3, 1, strides=1, kernel_initializer=truncateNormal, padding='same')(y)
    y = Activation('tanh')(y)
    return y