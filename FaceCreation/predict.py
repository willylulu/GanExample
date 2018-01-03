import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

class face_maker():
    
    def __init__(self):
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        self.generator = load_model('models/generator22.h5')
        
    def make_faces(self, batch):
        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
        fake_faces = self.generator.predict(noise)
        return (fake_faces/2+0.5)*255