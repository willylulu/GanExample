from keras.optimizers import Adam, RMSprop
from keras.layers import Input
from keras import backend as K
import tensorflow as tf

def gan_loss(inputshape, noiseshape, generator, discriminator, K, opt):
    realimg = Input(shape=imageshape)
    noise = Input(shape=noiseshape)
    fakeimg = generator(noise)
    d_real = discriminator(realimg)
    d_fake = discriminator(fakeimg)
    d_loss1 = K.mean(K.binary_crossentropy(K.ones_like(d_real), K.sigmoid(d_real)), axis=-1)
    d_loss2 = K.mean(K.binary_crossentropy(K.zeros_like(d_fake), K.sigmoid(d_fake)), axis=-1)
    d_loss = d_loss1 + d_loss2
    d_training_updates = opt.get_updates(discriminator.trainable_weights,[], d_loss)
    d_train = K.function([realimg, noise], [d_loss], d_training_updates)
    g_loss = K.mean(K.binary_crossentropy(K.ones_like(d_fake), K.sigmoid(d_fake)), axis=-1)
    g_training_updates = opt.get_updates(generator.trainable_weights,[], g_loss)
    g_train = K.function([noise], [g_loss], g_training_updates)
    
    return d_train, g_train

def lsgan_loss(inputshape, noiseshape, generator, discriminator, K, opt):
    realimg = Input(shape=imageshape)
    noise = Input(shape=noiseshape)
    fakeimg = generator(noise)
    d_real = discriminator(realimg)
    d_fake = discriminator(fakeimg)
    d_loss1 = tf.nn.l2_loss(K.ones_like(d_real) - d_real)
    d_loss2 = tf.nn.l2_loss(K.zeros_like(d_fake) - d_fake)
    d_loss = d_loss1 + d_loss2
    d_training_updates = opt.get_updates(discriminator.trainable_weights,[], d_loss)
    d_train = K.function([realimg, noise], [d_loss], d_training_updates)
    g_loss = tf.nn.l2_loss(K.ones_like(d_fake) - d_fake)
    g_training_updates = opt.get_updates(generator.trainable_weights,[], g_loss)
    g_train = K.function([noise], [g_loss], g_training_updates)
    
    return d_train, g_train

def wgan_loss(inputshape, noiseshape, generator, discriminator, K ):
    
    opt = RMSprop(lr=5e-5, clipvalue=0.01)
    
    realimg = Input(shape=imageshape)
    noise = Input(shape=noiseshape)
    fakeimg = generator(noise)
    d_real = discriminator(realimg)
    d_fake = discriminator(fakeimg)
    d_loss1 = K.mean(d_real, axis=-1)
    d_loss2 = K.mean(d_fake, axis=-1)
    d_loss = - d_loss1 + d_loss2
    d_training_updates = opt.get_updates(discriminator.trainable_weights,[], d_loss)
    d_train = K.function([realimg, noise], [d_loss], d_training_updates)
    g_loss =  - K.mean(d_fake, axis=-1)
    g_training_updates = opt.get_updates(generator.trainable_weights,[], g_loss)
    g_train = K.function([noise], [g_loss], g_training_updates)
    
    return d_train, g_train

def wgangp_loss(inputshape, noiseshape, generator, discriminator, K):
    
    opt = Adam(lr=1e-4, beta_1=0, beta_2=0.9)
    
    ϵ_input = K.placeholder(shape=(None,1,1,1))
    realimg = Input(shape=imageshape)
    noise = Input(shape=noiseshape)
    fakeimg = generator(noise)
    d_real = discriminator(realimg)
    d_fake = discriminator(fakeimg)
    d_loss1 = K.mean(d_real, axis=-1)
    d_loss2 = K.mean(d_fake, axis=-1)
    
    mixed_input = Input(shape=imageshape, tensor=ϵ_input * realimg + (1-ϵ_input) * fakeimg)
    
    grad_mixed = K.gradients(discriminator(mixed_input), [mixed_input])[0]
    norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
    grad_penalty = K.mean(K.square(norm_grad_mixed -1))
    
    d_loss = d_loss2 - d_loss1 + 10*grad_penalty
    d_training_updates = opt.get_updates(discriminator.trainable_weights,[], d_loss)
    d_train = K.function([realimg, noise, ϵ_input], [d_loss], d_training_updates)
    
    g_loss =  - K.mean(d_fake, axis=-1)
    g_training_updates = opt.get_updates(generator.trainable_weights,[], g_loss)
    g_train = K.function([noise], [g_loss], g_training_updates)
    
    return d_train, g_train