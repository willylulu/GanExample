import sys
import numpy as np
annos = open('list_attr_celeba.txt').readlines()

attrs = str.split(annos[1])
print(attrs)

new_attrs = ['5_o_Clock_Shadow', 'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Male', 'Mustache', 'Pale_Skin', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']
new_attrs_index = []
for x in new_attrs:
    new_attrs_index.append(attrs.index(x))
print(new_attrs_index)

annosAry = {}
for i in range(2,len(annos)):
    anno = str.split(annos[i])
    temp = [(int(i)+1)/2 for i in anno[1:]]
    temp2 = []
    for ii in new_attrs_index:
        temp2.append(temp[ii])
    annosAry[anno[0]] = temp2
    
print(annosAry["000001.jpg"])
print(len(annosAry["000001.jpg"]))

np.save("anno_dic.npy", annosAry)

img_list = open('image_list.txt').readlines()
imgIndex = [None]*len(img_list)

for i in range(1,len(img_list)):
    temp = str.split(img_list[i])
    imgIndex[int(temp[0])] = temp[2]
    
print(imgIndex[29999])

np.save("imgIndex.npy",imgIndex)