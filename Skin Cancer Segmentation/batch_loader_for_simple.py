################################# Libraries ################################

import os 
import matplotlib.pyplot as plt 
import numpy as np 
import random 
import segmentation_models as sm
from keras.utils import normalize


############################### Image Loader ################################

def load_image(img_dir, img_list):
    images=[]
    for i,image_name in enumerate(img_list):
        if (image_name.split(".")[1]=="npy"):
            image=np.load(img_dir + "/" + image_name)
            image=normalize(image , axis=1)
            images.append(image)

    images=np.array(images)
    return(images)


############################### Mask Loader #################################

def load_mask(img_dir, img_list):
    images=[]
    for i,image_name in enumerate(img_list):
        if (image_name.split(".")[1]=="npy"):
            image=np.load(img_dir + "/" + image_name)
            images.append(image)

    images=np.array(images)
    return(images)


############################### Batch Loader ################################

def image_loader_simple(img_dir, img_list, mask_dir, mask_list, batch_size):

    L=len(img_list)

    while True:
        batch_start = 0 
        batch_end = batch_size

        while batch_start < L:
            limit=min( batch_end , L)

            X=load_image(img_dir , img_list[batch_start:limit])
            Y=load_mask(mask_dir , mask_list[batch_start:limit])

            yield (X,Y)

            batch_start += batch_size
            batch_end += batch_size


############################# Test Batch Loader #############################

backbone="resnet34"
process_input=sm.get_preprocessing(backbone)

image_dir="C:/Datasets/Skin cancer segmentation/input_data/train/image"
mask_dir="C:/Datasets/Skin cancer segmentation/input_data/train/mask"

image_list=os.listdir(image_dir)
mask_list=os.listdir(mask_dir)
# print(len(image_list)  , len(mask_list))

batch_size=32
training_data=image_loader_simple(image_dir, image_list, mask_dir, mask_list, batch_size)

img,mask=training_data.__next__()


print(img.shape , mask.shape)

random_img=np.random.randint(0,batch_size -1)
test_img=img[random_img]
test_mask=mask[random_img]

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(test_img)
plt.title("image")
plt.subplot(1,2,2)
plt.imshow(test_mask)
plt.title("mask")
plt.show()

# print(np.unique(img))
# print(np.unique(mask[0]))

