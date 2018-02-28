import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import Input
from keras.optimizers import Adam, RMSprop
from keras.engine.topology import Layer
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from SingleGan import singleGan

class cycleGan():
    
    def __init__(self):
        self.batch = 1
        self.epochs = 300
        self.imageshape = (256, 256, 3)
        
        self.lr=2e-4
        
        self.x2y = singleGan()
        self.y2x = singleGan()
        self.x2y.summary()
        self.y2x.summary()
        self.trainingModel()
        
    def trainingModel(self):
        
        self.real_x = Input(shape=self.imageshape)
        self.real_y = Input(shape=self.imageshape)
        
        self.fake_x = self.y2x.generator([self.real_y])
        self.fake_y = self.x2y.generator([self.real_x])
        
        self.fake_xx = self.y2x.generator([self.fake_y])
        self.fake_yy = self.x2y.generator([self.fake_x])
        
        self.d_fake_x = self.y2x.discriminator([self.fake_x])
        self.d_fake_y = self.x2y.discriminator([self.fake_y])
        
        self.d_real_x = self.y2x.discriminator([self.real_x])
        self.d_real_y = self.x2y.discriminator([self.real_y])
        
        self.g_loss1 = K.mean(K.square(K.ones_like(self.d_fake_x) - K.sigmoid(self.d_fake_x)), axis=-1)
        self.g_loss2 = K.mean(K.square(K.ones_like(self.d_fake_y) - K.sigmoid(self.d_fake_y)), axis=-1)
        self.g_loss3 = K.mean(K.abs(self.real_x - self.fake_xx))
        self.g_loss4 = K.mean(K.abs(self.real_y - self.fake_yy))
        self.g_loss = self.g_loss1 + self.g_loss2 + 10 * self.g_loss3 + 10 * self.g_loss4
        
        self.d_loss1 = K.mean(K.square(K.ones_like(self.d_real_x) - K.sigmoid(self.d_real_x)), axis=-1)
        self.d_loss2 = K.mean(K.square(K.zeros_like(self.d_fake_x) - K.sigmoid(self.d_fake_x)), axis=-1)
        self.d_loss3 = K.mean(K.square(K.ones_like(self.d_real_y) - K.sigmoid(self.d_real_y)), axis=-1)
        self.d_loss4 = K.mean(K.square(K.zeros_like(self.d_fake_y) - K.sigmoid(self.d_fake_y)), axis=-1)
        self.d_loss = (self.d_loss1 + self.d_loss2 + self.d_loss3 + self.d_loss4)/2
    
    def set_lr(self, lr):
        
        self.d_training_updates = Adam(lr=lr, beta_1=0.5).get_updates(self.x2y.discriminator.trainable_weights + self.y2x.discriminator.trainable_weights,[], self.d_loss)
        self.d_train = K.function([self.real_x, self.real_y], [self.d_loss], self.d_training_updates)
        
        self.g_training_updates = Adam(lr=lr, beta_1=0.5).get_updates(self.x2y.generator.trainable_weights + self.y2x.generator.trainable_weights,[], self.g_loss)
        self.g_train = K.function([self.real_x, self.real_y], [self.g_loss], self.g_training_updates)
        
    def train(self, datasetx, datasety):
        
        print("training...")
        self.set_lr(self.lr)
        for k in range(0, self.epochs):
            
            seed = np.random.permutation(len(datasetx))
            
            ite=0
            while ite<len(datasetx) and ite<len(datasety):
                index = seed[ite:(ite+self.batch)]
                datax = datasetx[index]
                datay = datasety[index]
                
                for l in range(1):
                    errD, = self.d_train([datax, datay])
                    errD = np.mean(errD)
                for l in range(1):
                    errG, = self.g_train([datax, datay])
                    errG = np.mean(errG)
                print(errD, errG)
                ite+=self.batch
                
                if ite%100==0 and ite>0:
                    print("save")
                    pseed = np.random.randint(len(datasetx), size=8)
                    imagea = datasetx[pseed]
                    imageb = datasety[pseed]
                    fakey = self.x2y.generator.predict([imagea])
                    fakex = self.y2x.generator.predict([imageb])
                    fakeyy = self.x2y.generator.predict([fakex])
                    fakexx = self.y2x.generator.predict([fakey])
                    images = np.concatenate([imagea[:4], fakexx[:4], fakey[:4], imageb[:4], fakeyy[:4], fakex[:4], imagea[4:], fakexx[4:], fakey[4:], imageb[4:], fakeyy[4:], fakex[4:]])
                    width = 4
                    height = 6
                    new_im = Image.new('RGB', (256*height,256*width))
                    for ii in range(height):
                        for jj in range(width):
                            index=ii*width+jj
                            image = (images[index]/2+0.5)*255
                            image = image.astype(np.uint8)
                            new_im.paste(Image.fromarray(image,"RGB"), (256*ii,256*jj))
                    filename = "images/fakeFace%d.png"%(k/10)
                    new_im.save(filename)
                    self.x2y.generator.save("models/generator_x2y%d.h5"%(k/10))
                    self.y2x.generator.save("models/generator_y2x%d.h5"%(k/10))