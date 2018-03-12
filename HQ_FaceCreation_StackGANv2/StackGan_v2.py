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
from keras.backend.tensorflow_backend import set_session


class StackGan():
    
    def __init__(self, path, training=True):
        self.batch = 24
        self.imgNums = 30000
        self.epochs = 30000
        self.noiseNum = 100
        self.imageshape = [(64, 64, 3), (128, 128, 3), (256, 256, 3)]
        self.lr = 0.0002
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        if training:
            K.set_learning_phase(True)
            
            self.dataset = celebHq(path, self.batch, 30000)
            self.get_model()
            self.get_loss()
            self.get_optimizer()
        else:
            K.set_learning_phase(False)
            
            self.get_model()
            self.gnet_model.load_weights(path)
    
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
        
        self.d_loss64 = d_loss_64_real + d_loss_64_fake
        self.d_loss128 = d_loss_128_real + d_loss_128_fake
        self.d_loss256 = d_loss_256_real + d_loss_256_fake 
    
    def get_optimizer(self):
        
        self.g_training_updates = Adam(lr=self.lr, beta_1=0.5, beta_2=0.999).get_updates(self.gnet_model.trainable_weights,[], self.g_loss)
        self.g_train = K.function([self.gnet_noise], [self.g_loss], self.g_training_updates)
        
        self.d_training_updates64 = Adam(lr=self.lr, beta_1=0.5, beta_2=0.999).get_updates(self.dnet_64_model.trainable_weights,[], self.d_loss64)
        self.d_training_updates128 = Adam(lr=self.lr, beta_1=0.5, beta_2=0.999).get_updates(self.dnet_128_model.trainable_weights,[], self.d_loss128)
        self.d_training_updates256 = Adam(lr=self.lr, beta_1=0.5, beta_2=0.999).get_updates(self.dnet_256_model.trainable_weights,[], self.d_loss256)
        self.d_train64 = K.function([self.gnet_noise, self.real_64], [self.d_loss64], self.d_training_updates64)
        self.d_train128 = K.function([self.gnet_noise, self.real_128], [self.d_loss128], self.d_training_updates128)
        self.d_train256 = K.function([self.gnet_noise, self.real_256], [self.d_loss256], self.d_training_updates256)
        
    def train(self):
        
        print("training...")
        
        if os.path.exists('gnet_model.h5'):
            self.gnet_model.load_weights('gnet_model.h5')
        if os.path.exists('dnet_64_model.h5'):
            self.dnet_64_model.load_weights('dnet_64_model.h5')
        if os.path.exists('dnet_128_model.h5'):
            self.dnet_128_model.load_weights('dnet_128_model.h5')
        if os.path.exists('dnet_256_model.h5'):
            self.dnet_256_model.load_weights('dnet_256_model.h5')
        
        for ep in range(self.epochs):
            K.set_learning_phase(True)
            
            real_64, real_128, real_256 = self.dataset.getImages()
            noise = np.random.normal(0, 1.0, size=[self.batch, 100])
            for i in range(1):
                errD64 = self.d_train64([noise, real_64])
                errD128 = self.d_train128([noise, real_128])
                errD256 = self.d_train256([noise, real_256])
                errD = np.mean(errD64) + np.mean(errD128) + np.mean(errD256)
                
            for i in range(1):
                errG = self.g_train([noise])
                errG = np.mean(errG)
            
            print(errD, errG)
            
            if ep%10==0 and ep>0:
                K.set_learning_phase(False)
                
                noise = np.random.normal(0, 1.0, size=[16, 100])
                fake_64, fake_128, fake_256 = self.gnet_model.predict(noise)
                saveImages('./fake64', fake_64, 64, ep/100)
                saveImages('./fake128', fake_128, 128, ep/100)
                saveImages('./fake256', fake_256, 256, ep/100)
                print("save")
            if ep%100==0 and ep>0:
                self.gnet_model.save_weights('gnet_model.h5')
                self.dnet_64_model.save('dnet_64_model.h5')
                self.dnet_128_model.save('dnet_128_model.h5')                
                self.dnet_256_model.save('dnet_256_model.h5') 
    def predict(self, noise):
        
        fake_64, fake_128, fake_256 = self.gnet_model.predict(noise)
        return fake_64, fake_128, fake_256