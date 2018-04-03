import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.engine.topology import Layer
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from module import *
from skimage import io

class StarGan():
    
    def __init__(self, path):
        
        self.path = path
        self.lr = 2e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch = 8
        self.sample = 4
        self.epochs = 4e4
        self.imgSize = 256
        self.attSize = 16
        self.img_shape = (self.imgSize, self.imgSize, 3)
        self.att_shape = (self.attSize,)
        K.set_learning_phase(True)
        
        self.get_model()
        self.get_loss()
        self.get_optimizer()
    
    def get_model(self):
        
        self.img_input = Input(shape=self.img_shape)
        self.att_input = Input(shape=self.att_shape)
        
        self.img_output = generator(self.img_input, self.att_input, self.imgSize)
        
        self.g_model = Model(inputs=[self.img_input, self.att_input], outputs=[self.img_output])
        
        self.d_real_out= discriminator(self.img_input, self.att_input, self.imgSize, self.attSize)
        
        self.d_model = Model(inputs=[self.img_input, self.att_input], outputs=[self.d_real_out])
        
        print(self.g_model.summary())
        print(self.d_model.summary())
        
    def get_loss(self):
        
        self.img_a = Input(shape=self.img_shape)
        self.img_b = Input(shape=self.img_shape)

        self.att_a = Input(shape=self.att_shape)
        self.att_b = Input(shape=self.att_shape)                                                                                   
        self.img_a2b = self.g_model([self.img_a, self.att_b])
        self.img_a2b2a = self.g_model([self.img_a2b, self.att_a])
        
        d_a = self.d_model([self.img_a, self.att_a])
        d_a2b = self.d_model([self.img_a2b, self.att_b])
        d_w_img = self.d_model([self.img_b, self.att_a])
        d_w_att = self.d_model([self.img_a, self.att_b])
        
        d_loss_real = K.mean(K.square(K.ones_like(d_a) - d_a), axis=-1)
        d_loss_fake = K.mean(K.square(K.zeros_like(d_a2b) - d_a2b), axis=-1)
        d_loss_w_img = K.mean(K.square(K.zeros_like(d_w_img) - d_w_img), axis=-1)
        d_loss_w_att = K.mean(K.square(K.zeros_like(d_w_att) - d_w_att), axis=-1)
        
#         d_loss_real = K.mean(K.square(K.ones_like(d_img_a) - d_img_a), axis=-1)
#         d_loss_fake = K.mean(K.square(K.zeros_like(d_img_a2b) - d_img_a2b), axis=-1) 
        
#         d_loss_cls = K.mean(K.categorical_crossentropy(self.att_a, K.softmax(d_att_a)))
#         d_loss_cls = K.mean(K.square(self.att_a - K.softmax(d_att_a)))
    
        self.d_loss = d_loss_real + d_loss_fake + d_loss_w_img + d_loss_w_att
        
        g_loss_fake = K.mean(K.square(K.ones_like(d_a2b) - d_a2b), axis=-1)
#         g_loss_fake = K.mean(K.square(K.ones_like(d_img_a2b)- d_img_a2b), axis=-1)
        
#         g_loss_cls = K.mean(K.square(self.att_b - K.softmax(d_att_a2b)))
        
        g_loss_rec = K.mean(K.abs(self.img_a - self.img_a2b2a))
        
        self.g_loss = g_loss_fake + 10 * g_loss_rec
                          
    def get_optimizer(self):
        
        self.g_training_updates = Adam(lr=self.lr, decay=5e-9, beta_1=self.b1, beta_2=self.b2).get_updates(self.g_model.trainable_weights,[], self.g_loss)
        self.g_train = K.function([self.img_a, self.att_a, self.att_b], [self.g_loss], self.g_training_updates)
                          
        self.d_training_updates = Adam(lr=self.lr, decay=5e-9, beta_1=self.b1, beta_2=self.b2).get_updates(self.d_model.trainable_weights,[], self.d_loss)
        self.d_train = K.function([self.img_a, self.img_b, self.att_a, self.att_b], [self.d_loss], self.d_training_updates)
                          
    def train(self):
        
        print("load index")
        imgIndex = np.load("imgIndex.npy")
        imgAttr = np.load("anno_dic.npy").item()
        print("training")
        ite = 0
        for ep in range(int(self.epochs)):
            indexser = np.random.choice(len(imgIndex), self.batch*2)
            indexser1 = indexser[:self.batch]                 
            indexser2 = indexser[self.batch:]               
            img_as = [None]*self.batch
            img_bs = [None]*self.batch

            att_as = [None]*self.batch
            att_bs = [None]*self.batch

            for i in range(self.batch):

                temp_index = indexser1[i]
                img_fa = imgIndex[temp_index]
                while img_fa == None:
                    temp_index = np.random.choice(len(imgIndex), 1)[0]
                    img_fa = imgIndex[temp_index]
                att_a = imgAttr[img_fa]
                att_as[i] = att_a
                img_a = io.imread(os.path.join(self.path, str(temp_index)+".jpg"))
                img_a = img_a/127.5-1
                img_as[i] = img_a

                temp_index = indexser2[i]
                img_fb = imgIndex[temp_index]
                while img_fb == None:
                    temp_index = np.random.choice(len(imgIndex), 1)[0]
                    img_fb = imgIndex[temp_index]
                att_b = imgAttr[img_fb]
                att_bs[i] = att_b
                img_b = io.imread(os.path.join(self.path, str(temp_index)+".jpg"))
                img_b = img_b/127.5-1
                img_bs[i] = img_b

            img_as = np.array(img_as)
            img_bs = np.array(img_bs)
            att_as = np.array(att_as)
            att_bs = np.array(att_bs)

    #             noise = np.random.uniform(0, 1, self.batch)
    #             noise = np.expand_dims(noise, axis=-1)
    #             noise = np.expand_dims(noise, axis=-1)
    #             noise = np.expand_dims(noise, axis=-1)

            for i in range(1):
                errD = self.d_train([img_as, img_bs, att_as, att_bs])
            for i in range(1):
                errG = self.g_train([img_as, att_as, att_bs])
            print(np.mean(errD), np.mean(errG))    

            if ite%10==0 and ite>0:
                fakea2b = self.g_model.predict([img_as[:self.sample], att_bs[:self.sample]])
                fakea2a = self.g_model.predict([img_as[:self.sample], att_as[:self.sample]])
                fakea2b2a = self.g_model.predict([fakea2b[:self.sample], att_as[:self.sample]])
                images = np.concatenate([img_as[:self.sample], fakea2b, fakea2b2a, fakea2a], axis = 0)
                width = self.sample
                height = 4
                new_im = Image.new('RGB', (self.imgSize*height,self.imgSize*width))
                for ii in range(height):
                    for jj in range(width):
                        index=ii*width+jj
                        image = (images[index]/2+0.5)*255
                        image = image.astype(np.uint8)
                        new_im.paste(Image.fromarray(image,"RGB"), (self.imgSize*ii,self.imgSize*jj))
                filename = "img/fakeFace%d.png"%(ite//100)
                new_im.save(filename)
                self.g_model.save("model/generator%d.h5"%(ite//100))
            ite = ite + 1
    #             except KeyError:
    #                 print(indexs[0], img_fa, indexs[1], img_fb)



