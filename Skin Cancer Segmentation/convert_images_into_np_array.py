############################## Libraries ##############################

import matplotlib.pyplot as plt 
import numpy as np 
import glob 
import cv2 as cv 
from tqdm import tqdm 
from PIL import Image
import splitfolders 


######################## Conver images into .npy ######################

image_dataset = glob.glob("C:/Datasets/Skin cancer segmentation/images/*.jpg")

for i ,img in enumerate(tqdm(image_dataset)) :
    image=cv.imread(img )
    image=Image.fromarray(image) 
    image=image.resize((128,128))
    image=np.array(image)
    np.save("C:/Datasets/Skin cancer segmentation/numpy_arrays/image/image_" + str(i) + ".npy" , image)
    

######################## Conver masks into .npy #######################
    
mask_dataset = glob.glob("C:/Datasets/Skin cancer segmentation/masks/*.png")

for j , mask in enumerate(tqdm(mask_dataset)) :
    mask=cv.imread(mask,0)
    mask=Image.fromarray(mask) 
    mask=mask.resize((128,128))
    mask=np.array(mask)/255.0
    mask=np.expand_dims(mask, axis=2)
    np.save("C:/Datasets/Skin cancer segmentation/numpy_arrays/mask/mask_" + str(j) + ".npy" , mask)
    

############################# Sanity Check ############################  
    
img=np.load("C:/Datasets/Skin cancer segmentation/numpy_arrays/image/image_61.npy")
mask=np.load("C:/Datasets/Skin cancer segmentation/numpy_arrays/mask/mask_61.npy")
    
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(img)
plt.title("image")

plt.subplot(122)
plt.imshow(mask)
plt.title("mask")

plt.show()
    

############### Split folders for training and validation ###############

input_folder = 'C:/Datasets/Skin cancer segmentation/numpy_arrays'
output_folder = 'C:/Datasets/Skin cancer segmentation/input_data'

# Split with a ratio (.8, .2) for training and validation
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.8, .2), group_prefix=None)

