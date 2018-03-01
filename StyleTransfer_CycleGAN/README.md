# Style Transfer ( Cycle GAN
！[teaser](https://github.com/willylulu/GanExample/blob/master/StyleTransfer_CycleGAN/images/teaser.jpg?raw=true)
*	Transfer monet plant to photo and transfer photo to monet style
*	Using Cycle GAN (resnet or unet architecture
*	[Training dataset is from taesung_park/berkery](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

##	Usage
*	Pickle images to numpy array

`python3 preprocess.py <domain A dataset path> <domain B dataset path>`

*	Training model

`python3 train.py domainX.npy domainy.npy`

*	Predict photo

`python3 predict.py <domain A dataset path> <domain B dataset path>`

## Result
* First column is original monet paintings
* Second column is monet2photo image
* Third column is original photos
* Last column is photo2monet image
![test](https://github.com/willylulu/GanExample/blob/master/StyleTransfer_CycleGAN/test.jpg?raw=true)

## Reference
*	[XHUJOY/CycleGAN-tensorflow](https://github.com/XHUJOY/CycleGAN-tensorflow)
