# Face Attributes Transfer ( StarGAN
**Multiple domain interpretation ( style transfer**

**Change the face with different attributes ( hair color, gender, skin, facial expressions )**
![imga](https://github.com/willylulu/GanExample/blob/master/FaceAttributeChange_StarGAN/abc.png?raw=true)
The images from left to right are ( original image, black hair, blond hair, male, male with mustache, without smiling, pale face )

##	Usage
*	Dataset need to be preprocessed with following [this steps](https://github.com/willylulu/celeba-hq-modified)
	*	We use 256 x 256 celeba images as our training data
*	Preprocessing
	*	`python3 preprocessing.py`
*	Train
	*	`python3 train.py <path to celeba-hq/celeba-256>`
*	Predict
	*	Reference predict.ipynb to understand how to use it

##	Result
![imgb](https://github.com/willylulu/GanExample/blob/master/FaceAttributeChange_StarGAN/test1.png?raw=true)
*	The images from left to right are ( original image, black hair, brown hair, male, male with mustache, without smiling, pale face )
---
![imgc](https://github.com/willylulu/GanExample/blob/master/FaceAttributeChange_StarGAN/test2.png?raw=true)
*	The images from left to right are ( original image, black hair, blond hair, male, male with mustache, without smiling, pale face )
---
![imgd](https://github.com/willylulu/GanExample/blob/master/FaceAttributeChange_StarGAN/test3.png?raw=true)
*	The images from left to right are ( original image, without mustache, blond hair, female, more mustache, smiling, pale face )
---
![imge](https://github.com/willylulu/GanExample/blob/master/FaceAttributeChange_StarGAN/test4.png?raw=true)
*	The images from left to right are ( original image, blond hair, brown hair, female, more mustache, without smiling, pale face )

##	Reference
*	[yunjey/StarGAN](https://github.com/yunjey/StarGAN)
*	[arxiv/StarGAN](https://arxiv.org/abs/1711.09020)
*	[paarthneekhara/text-to-image](https://github.com/paarthneekhara/text-to-image)
*	[junyanz/CycleGAN](https://github.com/junyanz/CycleGAN)
