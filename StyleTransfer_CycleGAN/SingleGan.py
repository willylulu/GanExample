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
        self.imageshape = (256, 256, 3)

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
#             128 128 64
        self.gc2 = Conv2D(self.g_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb2 = InstanceNormalization2D()
#             64 64 128   
        self.gc3 = Conv2D(self.g_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb3 = InstanceNormalization2D()
#             32 32 256
        self.gc4 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb4 = InstanceNormalization2D()
#             16 16 512
        self.gc5 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb5 = InstanceNormalization2D()
#             8 8 512
        self.gc6 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb6 = InstanceNormalization2D()
#             4 4 512
        self.gc7 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb7 = InstanceNormalization2D()
#             2 2 512
        self.gc8 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb8 = InstanceNormalization2D()
#             1 1 512

        self.gc77 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb77 = InstanceNormalization2D()
#             2 2 512
        self.gc66 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb66 = InstanceNormalization2D()
#             4 4 512
        self.gc55 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb55 = InstanceNormalization2D()
#             8 8 512
        self.gc44 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb44 = InstanceNormalization2D()
#             16 16 512
        self.gc33 = Conv2DTranspose(self.g_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb33 = InstanceNormalization2D()
#             32 32 256
        self.gc22 = Conv2DTranspose(self.g_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb22 = InstanceNormalization2D()
#             64 64 128   
        self.gc11 = Conv2DTranspose(self.g_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb11 = InstanceNormalization2D()
#             128 128 64
        self.gc00 = Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             256 256 3

    def get_discriminator_weight(self):
#             discriminator weight
        self.dc0 = Conv2D(self.d_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             128 128 64
        self.dc1 = Conv2D(self.d_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             64 64 64
        self.dc2 = Conv2D(self.d_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             32 32 128
        self.dc3 = Conv2D(self.d_dim*8, 4, strides=1, padding='same', kernel_initializer=self.truncate_normal)
#             32 32 256
        self.dc4 = Conv2D(1, 1, strides=1, padding='same', kernel_initializer=self.truncate_normal)
        self.dd1 = Dense(1, kernel_initializer=self.truncate_normal)

    def get_generator_model(self):   
#             generator model
        self.ginput = Input(shape=self.imageshape)
        self.ng1 = self.gb1(self.gc1(self.ginput))
        self.ng2 = self.gb2(self.gc2(self.lk(self.ng1)))
        self.ng3 = self.gb3(self.gc3(self.lk(self.ng2)))
        self.ng4 = self.gb4(self.gc4(self.lk(self.ng3)))
        self.ng5 = self.gb5(self.gc5(self.lk(self.ng4)))
        self.ng6 = self.gb6(self.gc6(self.lk(self.ng5)))
        self.ng7 = self.gb7(self.gc7(self.lk(self.ng6)))
        
        self.ng8 = self.gb8(self.gc8(self.lk(self.ng7)))

        self.ng77 = self.ct([self.gb77(self.dp(self.gc77(self.rl(self.ng8)))), self.ng7])
        self.ng66 = self.ct([self.gb66(self.dp(self.gc66(self.rl(self.ng77)))), self.ng6])
        self.ng55 = self.ct([self.gb55(self.dp(self.gc55(self.rl(self.ng66)))), self.ng5])
        self.ng44 = self.ct([self.gb44(self.dp(self.gc44(self.rl(self.ng55)))), self.ng4])
        self.ng33 = self.ct([self.gb33(self.dp(self.gc33(self.rl(self.ng44)))), self.ng3])
        self.ng22 = self.ct([self.gb22(self.dp(self.gc22(self.rl(self.ng33)))), self.ng2])
        self.ng11 = self.ct([self.gb11(self.dp(self.gc11(self.rl(self.ng22)))), self.ng1])

        self.ng17 = self.tanh(self.gc00(self.rl(self.ng11)))

        self.generator = Model(inputs=[self.ginput], outputs=[self.ng17])

    def get_discriminator_model(self):  
#             discriminator model
        self.dinput = Input(shape=self.imageshape)
        self.nd0 = self.lk(self.dc0(self.dinput))
        self.nd1 = self.lk(self.dc1(self.nd0))
        self.nd2 = self.lk(self.dc2(self.nd1))
        self.nd3 = self.lk(self.dc3(self.nd2))
        self.nd4 = self.dc4(self.nd3)
        
#         self.nd5 = self.dd1(self.ft(self.nd4))

        self.discriminator = Model(inputs=[self.dinput], outputs=[self.nd4])
        
    def summary(self):
        self.generator.summary()
        self.discriminator.summary()