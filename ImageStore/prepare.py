import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# argv 1:target directory
# argv 2:target numpy matrix file
data = np.load(sys.argv[1]+sys.argv[2]+'.npy')
print(data.shape)
for i in range(len(data)):
    data[i]=cv2.GaussianBlur(data[i],(5,5),0)
np.save(sys.argv[2]+'GB.npy',data)
f,ax = plt.subplots(4,4)
for i in range(4):
    for j in range(4):
        index=i*4+j
        ax[i,j].imshow(data[index])
plt.show()
