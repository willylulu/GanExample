#	High quality face generator ( StackGAN++
*	Generate 256x256 high quality face images
*	Using StackGAN++ ( unconditional version
*	Dataset use celeba-hq provided by [progressive GAN](https://github.com/tkarras/progressive_growing_of_gans/tree/original-theano-version)

##	Usage
*	Dataset need to be preprocessed with following [this steps](https://github.com/willylulu/celeba-hq-modified)
*	Train

`python3 train.py <path to celeba-hq>`

*	Predict

`python3 predict.py <path to gnet model weights>`

##	Result
Random pick 32 images as input and train the model 200000 times
![imga](https://github.com/willylulu/GanExample/blob/master/HQ_FaceCreation_StackGANv2/fakefaces/face5.png?raw=true)

##  Improve training
To make more stable training and better quality, tune and modify code instead of original paper
*	Combine the 3 stage discriminator losses
*	Use he normal as initial weights
*	Use regulation and dropout
*	Decrease learning rate
*	Training discriminator one time and training generator many times until it converge in each loop
* Remove all images having bald or wearing hat or glasses attributes

##	Reference
*	[hanzhanggit/StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2)
*	[StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916)
*	[tensorflow/magenta/reviews](https://github.com/tensorflow/magenta/blob/master/magenta/reviews/GAN.md)
*	[tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans)
