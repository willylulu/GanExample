import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation, Reshape, Input, Dropout
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Concatenate
from keras.layers import BatchNormalization
from util import InstanceNormalization2D

class singleGan():
        
    def __init__(self):
        self.d_dim = 64
        self.g_dim = 64
        self.alpha = 0.2
        self.imageshape = (128, 128, 3)

        self.truncate_normal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
        self.random_normal = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)

        self.get_general_weight()
        self.get_generator_weight()
        self.get_discriminator_weight()
        self.get_generator_model()
        self.get_discriminator_model()


    def get_general_weight(self):
#             general
        self.lk = LeakyReLU(alpha=0.2)
        self.rl = Activation('relu')
        self.dp = Dropout(0.5)
        self.ft = Flatten()
        self.ct = Concatenate(axis=-1)
        self.tanh = Activation('tanh')

    def get_generator_weight(self):   
#             generator weight
        self.gc1 = Conv2D(self.g_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb1 = InstanceNormalization2D()
#             32 32 64
        self.gc2 = Conv2D(self.g_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb2 = InstanceNormalization2D()
#             16 16 128   
        self.gc3 = Conv2D(self.g_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb3 = InstanceNormalization2D()
#             8 8 256
        self.gc4 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb4 = InstanceNormalization2D()
#             4 4 512
        self.gc5 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb5 = InstanceNormalization2D()
#             2 2 512
        self.gc55 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb55 = InstanceNormalization2D()

        self.gc6 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb6 = InstanceNormalization2D()
#             1 1 512

        self.gc77 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb77 = InstanceNormalization2D()

        self.gc7 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb7 = InstanceNormalization2D()
#             2 2 512
        self.gc8 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb8 = InstanceNormalization2D()
#             4 4 512
        self.gc9 = Conv2DTranspose(self.g_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb9 = InstanceNormalization2D()
#             8 8 256
        self.gc10 = Conv2DTranspose(self.g_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb10 = InstanceNormalization2D()
#             16 16 128   
        self.gc11 = Conv2DTranspose(self.g_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb11 = InstanceNormalization2D()
#             32 32 64
        self.gc12 = Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)

    def get_discriminator_weight(self):
#             discriminator weight
        self.dc1 = Conv2D(self.d_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             32 32 64
        self.dc2 = Conv2D(self.d_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             16 16 128
        self.dc3 = Conv2D(self.d_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             8 8 256
        self.dc4 = Conv2D(self.d_dim*8, 4, strides=1, padding='same', kernel_initializer=self.truncate_normal)
#             4 4 512
        self.dc5 = Conv2D(1, 1, strides=1, padding='same', kernel_initializer=self.truncate_normal)
        self.dd1 = Dense(1, kernel_initializer=self.truncate_normal)

    def get_generator_model(self):   
#             generator model
        self.ginput = Input(shape=self.imageshape)
        self.ng1 = self.gb1(self.gc1(self.ginput))
        self.ng2 = self.gb2(self.gc2(self.lk(self.ng1)))
        self.ng3 = self.gb3(self.gc3(self.lk(self.ng2)))
        self.ng4 = self.gb4(self.gc4(self.lk(self.ng3)))
        self.ng5 = self.gb5(self.gc5(self.lk(self.ng4)))
        self.ng55 = self.gb55(self.gc55(self.lk(self.ng5)))

        self.ng6 = self.gb6(self.gc6(self.lk(self.ng55)))
        
        self.ng77 = self.dp(self.gc77(self.rl(self.ng6)))
        self.ng88 = self.ct([self.gb77(self.ng77), self.ng55])

        self.ng7 = self.dp(self.gc7(self.rl(self.ng88)))
        self.ng8 = self.ct([self.gb7(self.ng7), self.ng5])

        self.ng9 = self.dp(self.gc8(self.rl(self.ng8)))
        self.ng10 = self.ct([self.gb8(self.ng9), self.ng4])

        self.ng11 = self.dp(self.gc9(self.rl(self.ng10)))
        self.ng12 = self.ct([self.gb9(self.ng11), self.ng3])  

        self.ng13 = self.dp(self.gc10(self.rl(self.ng12)))
        self.ng14 = self.ct([self.gb10(self.ng13), self.ng2])  

        self.ng15 = self.dp(self.gc11(self.rl(self.ng14)))
        self.ng16 = self.ct([self.gb11(self.ng15), self.ng1])

        self.ng17 = self.tanh(self.gc12(self.rl(self.ng16)))

        self.generator = Model(inputs=[self.ginput], outputs=[self.ng17])

    def get_discriminator_model(self):  
#             discriminator model
        self.dinput = Input(shape=self.imageshape)
        self.nd1 = self.lk(self.dc1(self.dinput))
        self.nd2 = self.lk(self.dc2(self.nd1))
        self.nd3 = self.lk(self.dc3(self.nd2))
        self.nd4 = self.lk(self.dc4(self.nd3))
        self.nd5 = self.dc5(self.nd4)
        
#         self.nd5 = self.dd1(self.ft(self.nd4))

        self.discriminator = Model(inputs=[self.dinput], outputs=[self.nd5])
        
    def summary(self):
        self.generator.summary()
        self.discriminator.summary()