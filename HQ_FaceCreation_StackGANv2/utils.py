import os
import numpy as np
from skimage import io
from PIL import Image

class celebHq():
    
    def __init__(self, path, batch, imgNums):
        self.start = 0
        self.batch = batch
        self.imgNums = imgNums
        self.paths = [os.path.join(path, 'celeba-64'), os.path.join(path, 'celeba-128'), os.path.join(path, 'celeba-256')]
        
    def getImages(self):
        
        if self.start + self.batch > self.imgNums:
            self.start = 0
        
        celeba64 = [None]*self.batch
        celeba128 = [None]*self.batch
        celeba256 = [None]*self.batch
        j = 0
        for i in range(self.start, self.start + self.batch):
            celeba64[j] = io.imread(os.path.join(self.paths[0], str(i)+'.jpg'))/127.5-1
            celeba128[j] = io.imread(os.path.join(self.paths[1], str(i)+'.jpg'))/127.5-1
            celeba256[j] = io.imread(os.path.join(self.paths[2], str(i)+'.jpg'))/127.5-1
            j += 1
        
        celeba64 = np.array(celeba64)
        celeba128 = np.array(celeba128)
        celeba256 = np.array(celeba256)
        
        self.start = self.start + self.batch
        
        return celeba64, celeba128, celeba256
    
def saveImages(dpath, images, img_size, k):
    
    width = 4
    height = 4
    new_im = Image.new('RGB', (img_size*height,img_size*width))
    for ii in range(height):
        for jj in range(width):
            index=ii*width+jj
            image = (images[index]/2+0.5)*255
            image = image.astype(np.uint8)
            new_im.paste(Image.fromarray(image,"RGB"), (img_size*ii,img_size*jj))
    filename = os.path.join(dpath,"fakeFace%d.png"%k)
    new_im.save(filename)