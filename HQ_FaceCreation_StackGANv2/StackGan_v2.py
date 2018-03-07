import os
import random
import numpy as np
from utils import *
from module import *

import tensorflow as tf
from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.engine.topology import Layer
from keras import backend as K

class StackGan():
    
    def __init__(self, path):
        self.batch = 64
        self.imgNums = 30000
        self.epochs = 30000
        self.noiseNum = 100
        self.imageshape = [(64, 64, 3), (128, 128, 3), (256, 256, 3)]
        self.lr = 0.0002
        self.dataset = celebHq(path, 64, 30000)
    
    def get_model(self):
        self.d_64_input = Input(shape=self.imageshape[0])
        self.d_128_input = Input(shape=self.imageshape[1])
        self.d_256_input = Input(shape=self.imageshape[2])
        
        self.dnet_64 = dnet_64(self.d_64_input)
        self.dnet_128 = dnet_128(self.d_128_input)
        self.dnet_256 = dnet_256(self.d_256_input)
        
        self.dnet_64_model = Model(inputs=[self.d_64_input], outputs=[self.dnet_64])
        self.dnet_128_model = Model(inputs=[self.d_128_input], outputs=[self.dnet_128])        
        self.dnet_256_model = Model(inputs=[self.d_256_input], outputs=[self.dnet_256])
        
        self.g_input = Input(shape=[self.noiseNum,])
        
        self.gnet_64 = gnet_64(self.g_input)
        self.gnet_128 = gnet_128(self.gnet_64)
        self.gnet_256 = gnet_256(self.gnet_128)
        
        self.gnet_64_output = gnet_64_output(self.gnet_64)
        self.gnet_128_output = gnet_64_output(self.gnet_128)
        self.gnet_256_output = gnet_64_output(self.gnet_256)
        
        self.gnet_model = Model(inputs=[self.g_input], outputs=[self.gnet_64_output, self.gnet_128_output, self.gnet_256_output])
        
        print(self.gnet_model.summary())
        print(self.dnet_64_model.summary())
        print(self.dnet_128_model.summary())
        print(self.dnet_256_model.summary())
        
        
    def get_loss(self):
        
        self.gnet_noise = Input(shape=[100,])
        fake_64, fake_128, fake_256 = self.gnet_model([self.gnet_noise])
        
        fake_64_mean, fake_64_var = tf.nn.moments(fake_64, [-3,-2,-1])
        fake_128_mean, fake_128_var = tf.nn.moments(fake_128, [-3,-2,-1])
        fake_256_mean, fake_256_var = tf.nn.moments(fake_256, [-3,-2,-1])
        
        mean_64_128_loss = K.square(fake_64_mean - fake_128_mean)
        mean_128_256_loss = K.square(fake_128_mean - fake_256_mean)
        
        var_64_128_loss = K.square(fake_64_var - fake_128_var)
        var_128_256_loss = K.square(fake_128_var - fake_256_var)
        
        color_consistancy = mean_64_128_loss + mean_128_256_loss + var_64_128_loss + var_128_256_loss
        color_consistancy = tf.expand_dims(color_consistancy, -1)
        color_consistancy = tf.expand_dims(color_consistancy, -1)
        
        dnet_64_fake = self.dnet_64_model(fake_64)
        dnet_128_fake = self.dnet_128_model(fake_128)
        dnet_256_fake = self.dnet_256_model(fake_256)
        
        g_loss_64 = K.mean(K.binary_crossentropy(K.ones_like(dnet_64_fake), K.sigmoid(dnet_64_fake)), axis=-1)
        g_loss_128 = K.mean(K.binary_crossentropy(K.ones_like(dnet_128_fake), K.sigmoid(dnet_128_fake)), axis=-1)
        g_loss_256 = K.mean(K.binary_crossentropy(K.ones_like(dnet_256_fake), K.sigmoid(dnet_256_fake)), axis=-1)
        
        self.g_loss = g_loss_64 + g_loss_128 + g_loss_256 + 50*color_consistancy
        
        self.real_64 = Input(shape=[64, 64, 3])
        self.real_128 = Input(shape=[128, 128, 3])
        self.real_256 = Input(shape=[256, 256, 3])
        
        dnet_64_real = self.dnet_64_model(self.real_64)
        dnet_128_real = self.dnet_128_model(self.real_128)
        dnet_256_real = self.dnet_256_model(self.real_256)
        
        
        d_loss_64_real = K.mean(K.binary_crossentropy(K.ones_like(dnet_64_real), K.sigmoid(dnet_64_real)), axis=-1)
        d_loss_64_fake = K.mean(K.binary_crossentropy(K.zeros_like(dnet_64_fake), K.sigmoid(dnet_64_fake)), axis=-1)
        
        d_loss_128_real = K.mean(K.binary_crossentropy(K.ones_like(dnet_128_real), K.sigmoid(dnet_128_real)), axis=-1)
        d_loss_128_fake = K.mean(K.binary_crossentropy(K.zeros_like(dnet_128_fake), K.sigmoid(dnet_128_fake)), axis=-1)
        
        d_loss_256_real = K.mean(K.binary_crossentropy(K.ones_like(dnet_256_real), K.sigmoid(dnet_256_real)), axis=-1)
        d_loss_256_fake = K.mean(K.binary_crossentropy(K.zeros_like(dnet_256_fake), K.sigmoid(dnet_256_fake)), axis=-1)
        
        self.d_loss = d_loss_64_real + d_loss_64_fake + d_loss_128_real + d_loss_128_fake + d_loss_256_real + d_loss_256_fake 
    
    def get_optimizer(self):
        
        self.g_training_updates = Adam(lr=self.lr, beta_1=0.5, beta_2=0.9).get_updates(self.gnet_model.trainable_weights,[], self.g_loss)
        self.g_train = K.function([self.gnet_noise], [self.g_loss], self.g_training_updates)
        
        self.d_training_updates = Adam(lr=self.lr, beta_1=0.5, beta_2=0.9).get_updates(self.dnet_64_model.trainable_weights + self.dnet_128_model.trainable_weights + self.dnet_256_model.trainable_weights,[], self.d_loss)
        self.d_train = K.function([self.gnet_noise, self.real_64, self.real_128, self.real_256], [self.d_loss], self.d_training_updates)
        
    def train(self):
        
        self.get_model()
        self.get_loss()
        self.get_optimizer()
        
        print("training...")
        
        for ep in range(self.epochs):
            real_64, real_128, real_256 = self.dataset.getImages()
            noise = np.random.normal(0, 1.0, size=[self.batch, 100])
            for i in range(1):
                errD = self.d_train([noise, real_64, real_128, real_256])
            for i in range(2):
                errG = self.g_train([noise])
            
            print(np.mean(errD), np.mean(errG))
            
            if ep%10==0 and ep>0:
                noise = np.random.normal(0, 1.0, size=[16, 100])
                fake_64, fake_128, fake_256 = self.gnet_model.predict(noise)
                saveImages('./fake64', fake_64, 64, ep/100)
                saveImages('./fake128', fake_128, 128, ep/100)
                saveImages('./fake256', fake_256, 256, ep/100)