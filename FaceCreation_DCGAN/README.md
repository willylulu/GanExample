# Face Creation ( DCGAN

##	Using Generative Adversarial Network generates faces 
![github](https://github.com/willylulu/GanExample/blob/master/FaceCreation_DCGAN/test.png "image")

##	Dataset
*	The GAN model trained by celebA dataset
*	http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

##	Libary used
* tensorflow
*	keras (Tensorflow backend
*	numpy
*	matplotlib

##  Prepare
### Train
* You need to put all image in numpy matrix
* numpy matrix needs to be (-1,64,64,3)
### Predict
* The output shape is (-1,64,64,3) 
##  Use Example
```	python line-numbers
from predict import face_maker
path = '<path to predict model>'
fm = face_maker(path)
faces = fm.make_faces(64)
#make 64 fake faces
#shape is (64, 64, 64, 3)
#value type is int, RGB(255, 255, 255)
```
