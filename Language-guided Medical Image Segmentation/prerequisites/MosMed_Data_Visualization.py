import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
import numpy as np 
import glob

train_img_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/train/images"
train_mask_store_dir = "/home/asosoft/Seivan/MosMedData+_prepared/train/masks"

image_dataset = sorted(glob.glob(train_img_store_dir + "/*"))
mask_dataset = sorted(glob.glob(train_mask_store_dir + "/*"))

rand_num = np.random.randint(0, len(image_dataset)-1)
plt.figure(figsize=(12,12))
plt.subplot(421)
plt.imshow(Image.open(image_dataset[rand_num]), cmap="grey")
plt.subplot(422)
plt.imshow(Image.open(mask_dataset[rand_num]), cmap="grey")

plt.subplot(423)
plt.imshow(Image.open(image_dataset[rand_num+1]), cmap="grey")
plt.subplot(424)
plt.imshow(Image.open(mask_dataset[rand_num+1]), cmap="grey")

plt.subplot(425)
plt.imshow(Image.open(image_dataset[rand_num+2]), cmap="grey")
plt.subplot(426)
plt.imshow(Image.open(mask_dataset[rand_num+2]), cmap="grey")

plt.subplot(427)
plt.imshow(Image.open(image_dataset[rand_num+3]), cmap="grey")
plt.subplot(428)
plt.imshow(Image.open(mask_dataset[rand_num+3]), cmap="grey")

plt.savefig("train.png")




