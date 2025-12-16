import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
import glob
import shutil


image_dir = "F:/Visual Language Grounding (VLG)/Datasets/QaTa-Covid2019/QaTa-COV19/QaTa-COV19-v2/Train Set/Images/"
mask_dir = "F:/Visual Language Grounding (VLG)/Datasets/QaTa-Covid2019/QaTa-COV19/QaTa-COV19-v2/Train Set/Ground-truths/"

train_img_store_dir = "C:/Datasets/QaTa_prepared/train/image"
train_mask_store_dir = "C:/Datasets/QaTa_prepared/train/mask"

image_list = pd.read_excel("F:/Visual Language Grounding (VLG)/Datasets/QaTa-Covid2019/Train_ID.xlsx")["Image"].tolist()

for img in image_list:
    shutil.copy(image_dir + img.split("mask_")[1], train_img_store_dir)
    
    shutil.copy(mask_dir + img, train_mask_store_dir)
    
    

image_dataset = sorted(glob.glob(train_img_store_dir + "/*"))
mask_dataset = sorted(glob.glob(train_mask_store_dir + "/*"))

rand_num = np.random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(Image.open(image_dataset[rand_num]), cmap="grey")
plt.subplot(122)
plt.imshow(Image.open(mask_dataset[rand_num]), cmap="grey")
plt.show()



val_img_store_dir = "C:/Datasets/QaTa_prepared/val/image"
val_mask_store_dir = "C:/Datasets/QaTa_prepared/val/mask"

val_image_list = pd.read_excel("F:/Visual Language Grounding (VLG)/Datasets/QaTa-Covid2019/Val_ID.xlsx")["Image"].tolist()

for img in val_image_list:
    shutil.copy(image_dir + img.split("mask_")[1], val_img_store_dir)
    
    shutil.copy(mask_dir + img, val_mask_store_dir)
    
    
image_dataset = sorted(glob.glob(val_img_store_dir + "/*"))
mask_dataset = sorted(glob.glob(val_mask_store_dir + "/*"))

rand_num = np.random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(Image.open(image_dataset[rand_num]), cmap="grey")
plt.subplot(122)
plt.imshow(Image.open(mask_dataset[rand_num]), cmap="grey")
plt.show()






















 