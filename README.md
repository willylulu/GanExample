# Img2Img Image Storat8ion

##	Make blur image recovery to origin image
![github](https://i.imgur.com/0LMT0nY.jpg "image")
* Middle image is origin image
* Right image is blur image using guassian blur
* Left image is recovered image using Generative Adversarial Network

##	Dataset
*	The GAN model trained by wiki-face dataset
*	https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
*	We use caltech 101 for testing
*	http://www.vision.caltech.edu/Image_Datasets/Caltech101/

##	Libary used
*	keras (Tensorflow backend
*	numpy
*	cv2
*	matplotlib

##  Prepare
* You need to put all image in numpy matrix
* numpy matrix needs to be (-1,64,64,3)
