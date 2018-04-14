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
<img src="https://github.com/willylulu/GanExample/blob/master/HQ_FaceCreation_StackGANv2/fakefaces/face1.png?raw=true" width="256" height="256">
<img src="https://github.com/willylulu/GanExample/blob/master/HQ_FaceCreation_StackGANv2/fakefaces/face2.png?raw=true" width="256" height="256">

<img src="https://github.com/willylulu/GanExample/blob/master/HQ_FaceCreation_StackGANv2/fakefaces/face3.png?raw=true" width="256" height="256">
<img src="https://github.com/willylulu/GanExample/blob/master/HQ_FaceCreation_StackGANv2/fakefaces/face4.png?raw=true" width="256" height="256">

##	Reference
*	[hanzhanggit/StackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2)
*	[StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916)
