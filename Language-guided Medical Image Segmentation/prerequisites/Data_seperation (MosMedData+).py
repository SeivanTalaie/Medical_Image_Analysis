import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
import glob
import shutil


image_dir = "/home/asosoft/Seivan/MosMedData+/2/frames/"
mask_dir = "/home/asosoft/Seivan/MosMedData+/2/masks/"

train_img_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/train/images"
train_mask_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/train/masks"

image_list = pd.read_csv("/home/asosoft/Seivan/MosMedData+/Train_text_MosMedData+ 1(in).csv")["Image"].tolist()

for img in image_list:
    shutil.copy(image_dir + img, train_img_store_dir)
    
    shutil.copy(mask_dir + img, train_mask_store_dir)
    
    

image_dataset = sorted(glob.glob(train_img_store_dir + "/*"))
mask_dataset = sorted(glob.glob(train_mask_store_dir + "/*"))

rand_num = np.random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(Image.open(image_dataset[rand_num]), cmap="grey")
plt.subplot(122)
plt.imshow(Image.open(mask_dataset[rand_num]), cmap="grey")
plt.savefig("train.png")



val_img_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/val/images"
val_mask_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/val/masks"

val_image_list = pd.read_csv("/home/asosoft/Seivan/MosMedData+/Val_text_MosMedData+ 1(in).csv")["Image"].tolist()

for img in val_image_list:
    shutil.copy(image_dir + img, val_img_store_dir)
    
    shutil.copy(mask_dir + img, val_mask_store_dir)
    
    
image_dataset = sorted(glob.glob(val_img_store_dir + "/*"))
mask_dataset = sorted(glob.glob(val_mask_store_dir + "/*"))

rand_num = np.random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(Image.open(image_dataset[rand_num]), cmap="grey")
plt.subplot(122)
plt.imshow(Image.open(mask_dataset[rand_num]), cmap="grey")
plt.savefig("val.png")



test_img_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/test/images"
test_mask_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/test/masks"

test_image_list = pd.read_csv("/home/asosoft/Seivan/MosMedData+/Test_text_MosMedData+(in).csv")["Image"].tolist()

for img in test_image_list:
    shutil.copy(image_dir + img, test_img_store_dir)
    
    shutil.copy(mask_dir + img, test_mask_store_dir)
    
    
image_dataset = sorted(glob.glob(test_img_store_dir + "/*"))
mask_dataset = sorted(glob.glob(test_mask_store_dir + "/*"))

rand_num = np.random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(Image.open(image_dataset[rand_num]), cmap="grey")
plt.subplot(122)
plt.imshow(Image.open(mask_dataset[rand_num]), cmap="grey")
plt.savefig("test.png")



















 