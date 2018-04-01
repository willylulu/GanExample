import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import LeakyReLU, Activation, Input, Reshape, Flatten, Dense
from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D, Lambda, Concatenate, Add
from keras.layers import BatchNormalization
from ops import *
from util import InstanceNormalization2D

# truncate_normal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)

def generator(img, attr, size):
    
    def tileAttr(x):
        x = tf.expand_dims(x, axis = 1)
        x = tf.expand_dims(x, axis = 2)
        return tf.tile(x, [1, size, size, 1])
    
    y = Concatenate()([img, Lambda(tileAttr)(attr)])
    
    y = Conv2D(64, 7, padding="same", kernel_initializer='he_normal' )(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(128, 4, strides=2, padding="same", kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(256, 4, strides=2, padding="same", kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)
    
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    y = residual_block(y, 256, 3)
    
    y = Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)
    
    y = Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = InstanceNormalization2D()(y)
    y = Activation('relu')(y)
    
    y = Conv2D(3, 7, strides=1, padding='same', kernel_initializer='he_normal')(y)
    y = Activation('tanh')(y)
    return y

def discriminator(img, attr, size, att_size):
    
    def tileAttr2(x):
        x = tf.expand_dims(x, axis = 1)
        x = tf.expand_dims(x, axis = 2)
        return tf.tile(x, [1, size//64, size//64, 1])
    
    y = Conv2D(64, 4, strides=2, padding='same', kernel_initializer='he_normal')(img)
    y = LeakyReLU(alpha=0.01)(y) # 64 64 64
    
    y = Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = LeakyReLU(alpha=0.01)(y) # 32 32 128
    
    y = Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = LeakyReLU(alpha=0.01)(y) # 16 16 256
    
    y = Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = LeakyReLU(alpha=0.01)(y) # 8 8 512
    
    y = Conv2D(1024, 4, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = LeakyReLU(alpha=0.01)(y) # 4 4 1024
    
    y = Conv2D(2048, 4, strides=2, padding='same', kernel_initializer='he_normal')(y)
    y = LeakyReLU(alpha=0.01)(y) # 2 2 2048
    
    y = Concatenate()([y, Lambda(tileAttr2)(attr)])
    
    y = Conv2D(2048, 1, strides=1, kernel_initializer='he_normal')(y)
    y = LeakyReLU(alpha=0.01)(y) # 2 2 2048
    
    y = Conv2D(1, 1, kernel_initializer='he_normal')(y)
    
    return y