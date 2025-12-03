import os 
import matplotlib.pyplot as plt 
import numpy as np 
import random 
from keras.utils import normalize


def load_image(img_dir , img_list):
    images=[]
    for i,image_name in enumerate(img_list):
        if (image_name.split(".")[1]=="npy"):
            image=np.load(img_dir + "/" + image_name)
            images.append(image)

    images=np.array(images)
    return(images)

def image_loader(img_dir , img_list , mask_dir , mask_list , batch_size):

    L=len(img_list)

    while True:
        batch_start = 0 
        batch_end = batch_size

        while batch_start < L:
            limit=min( batch_end , L)

            X=load_image(img_dir , img_list[batch_start:limit])
            X=np.reshape(X , (-1 , 128,128,3))
            X=normalize(X , axis=1)
            Y=load_image(mask_dir , mask_list[batch_start:limit])
            Y=np.reshape(Y,(-1 , 128,128,4))
            yield (X,Y)

            batch_start += batch_size
            batch_end += batch_size
            
            
########### test the data generator ########### 

# image_dir="C:\Datasets/Brats2020/Input_data_2D/train\image"
# mask_dir="C:\Datasets/Brats2020/Input_data_2D/train\mask"
# image_list=os.listdir(image_dir)
# mask_list=os.listdir(mask_dir)
# # print(len(image_list)  , len(mask_list))
# batch_size=1
# training_data=image_loader(image_dir,image_list , mask_dir , mask_list , batch_size)


# img,mask=training_data.__next__()

# print(img.shape   ,  mask.shape)

# random_img=np.random.randint(0,img.shape[0]-1)
# test_img=img[39]
# test_mask=mask[39]
# test_mask=np.argmax(test_mask , axis=2)

# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.imshow(test_img , cmap="gray")
# plt.title("image")
# plt.subplot(1,2,2)
# plt.imshow(test_mask, cmap="gray")
# plt.title("mask")
# plt.show()


            
            
            
            
            
            
            
            
            
            
            