from keras.engine.topology import Layer
import tensorflow as tf

class InstanceNormalization2D(Layer):
    
    def __init__(self, **kwargs):
        super(InstanceNormalization2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[3],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[3],), initializer="zero", trainable=True)
        super(InstanceNormalization2D, self).build(input_shape)

    def call(self, x, mask=None):
        mean, variance = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.shift

    def compute_output_shape(self, input_shape):
        return input_shape