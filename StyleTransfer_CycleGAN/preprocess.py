import os
import sys
import numpy as np
from skimage import io, transform

pathA = sys.argv[1]
pathB = sys.argv[2]

filesA = os.listdir(pathA)
fileAnum = len(filesA)
filesB = os.listdir(pathB)
fileBnum = len(filesB)

x = [None]*fileAnum
y = [None]*fileBnum

for i in range(fileAnum):
    x[i] = transform.resize(io.imread(os.path.join(pathA, filesA[i])),(128, 128))*2-1
x = np.array(x)
print(np.min(x))
print(x.shape)
for i in range(fileBnum):
    y[i] = transform.resize(io.imread(os.path.join(pathB, filesB[i])),(128, 128))*2-1
y = np.array(y)
print(np.min(y))
print(y.shape)

np.save('domainX.npy', x)
np.save('domainy.npy', y)