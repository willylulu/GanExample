# GAN Loss Library

### GAN Losses implemented by Keras

*	Original GAN Loss
*	Least Square Loss
*	Wasserstein GAN
*	Wasserstein GAN Gradient Penalty

##	Import
```	python line-numbers  
parentPath = os.path.abspath("<path to GanLossLib Directory>")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from ganlosslib import *
```   
##	Function
```	python line-numbers  
gan_loss(inputshape, noiseshape, generator, discriminator, K, opt)
lsgan_loss(inputshape, noiseshape, generator, discriminator, K, opt)
wgan_loss(inputshape, noiseshape, generator, discriminator, K)
wgangp_loss(inputshape, noiseshape, generator, discriminator, K)
``` 
*	inputshape
	*	The data shape input for discriminator real data.
*	noiseshape
	*	The data shape input for generator noise input data.
*	generator
	*	generator Keras model
*	discriminator
	*	discriminator Keras model
*	K
	*	keras backend
*	opt
	*	optimizer

##	WGAN & WGANGP
Following the paper, we use the optimizer as same as the paper.

##	Requirement
*	keras
*	tensorflow

##	Todo
*	Add conditional input for implement conditional GAN Loss

##	Reference
*	https://github.com/tjwei/GANotebooks
*	https://github.com/carpedm20/DCGAN-tensorflow
*	https://github.com/GunhoChoi/LSGAN-TF
