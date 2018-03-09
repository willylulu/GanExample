import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation, Input
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Concatenate, Add, UpSampling2D
from keras.layers import BatchNormalization
from keras import backend as K

truncateNormal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)

def dnet_basic(x, d_dim):
    
    y = Conv2D(d_dim, 4, strides=2, padding='same', kernel_initializer=truncateNormal)(x)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(d_dim*2, 4, strides=2, padding='same', kernel_initializer=truncateNormal)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(d_dim*4, 4, strides=2, padding='same', kernel_initializer=truncateNormal)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Conv2D(d_dim*8, 4, strides=2, padding='same', kernel_initializer=truncateNormal)(y)
    y = LeakyReLU(alpha=0.2)(y)
    return y

def dnet_conv(x, d_dim , output_dim):
    
    y = Conv2D(output_dim, 3, strides=2, padding='same', kernel_initializer=truncateNormal)(x) # 4 4 1024
    y = LeakyReLU(alpha=0.2)(y)
    return y

def dnet_block3x3(x, d_dim, output_dim):
    
    y = Conv2D(output_dim, 3, strides=1, padding='same', kernel_initializer=truncateNormal)(x)
    y = LeakyReLU(alpha=0.2)(y)
    return y

def gnet_upsample(y, g_dim):
#     y = Conv2DTranspose(g_dim, 3, strides=2, kernel_initializer=truncateNormal, padding='same')(y)
    y = UpSampling2D(size=(2,2))(y)
    y = Conv2D(g_dim, 3, strides=1, padding='same', kernel_initializer=truncateNormal)(y)
    y = BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    y = Lambda(glu)(y) #128, 128, 32
    return y

def residual_block(x, dim):
    y = Conv2D(dim*2, 3, strides=1, padding='same', kernel_initializer=truncateNormal)(x)
    y = BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    y = Lambda(glu)(y)
    y = Conv2D(dim, 3, strides=1, padding='same', kernel_initializer=truncateNormal)(y)
    y = BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    return Add()([x,y])

def glu(x):
    channel = K.int_shape(x)[-1]
    channel = channel//2
    a = x[..., :channel]
    b = x[..., channel:]
    return a * K.sigmoid(b)
