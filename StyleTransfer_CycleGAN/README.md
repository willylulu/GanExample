# Style Transfer ( Cycle GAN
*	Transfer monet plant to photo and transfer photo to monet style
*	Using Cycle GAN (unet architecture
*	[Training dataset is from taesung_park/berkery](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

##	Usage
*	Pickle images to numpy array

`python3 preprocess.py <domain A dataset path> <domain B dataset path>`

*	Training model

`python3 train.py domainX.npy domainy.npy`

*	Predict photo

`python3 predict.py <domain A dataset path> <domain B dataset path>`

## Reference
*	[XHUJOY/CycleGAN-tensorflow](https://github.com/XHUJOY/CycleGAN-tensorflow)