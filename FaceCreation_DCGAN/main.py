import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from predict import face_maker

fm = face_maker()

fake_faces = fm.make_faces(64)

width = 8
new_im = Image.new('RGB', (64*width,64*width))
for ii in range(width):
    for jj in range(width):
        index=ii*width+jj
        image = fake_faces[index]
        image = image.astype(np.uint8)
        new_im.paste(Image.fromarray(image,"RGB"), (64*ii,64*jj))
new_im.save('test.png')