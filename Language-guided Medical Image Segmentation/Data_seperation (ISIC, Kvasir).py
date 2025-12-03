import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

############################## ISIC 2016 ##############################

isic_train = "F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/isic/anns/train.json"
isic_test = "F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/isic/anns/test.json"
isic_val = "F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/isic/anns/val.json"

isic_train_json = pd.read_json(isic_train)
isic_test_json = pd.read_json(isic_test)
isic_val_json = pd.read_json(isic_val)

isic_train_img_list = list(isic_train_json.img_name)
isic_test_img_list = list(isic_test_json.img_name)
isic_val_img_list = list(isic_val_json.img_name)

isic_train_mask_list = list(isic_train_json.mask_name)
isic_test_mask_list = list(isic_test_json.mask_name)
isic_val_mask_list = list(isic_val_json.mask_name)


unified_img_dir = "F:/Visual Language Grounding (VLG)/Datasets/ISIC 2016/unified_images/"
unified_mask_dir = "F:/Visual Language Grounding (VLG)/Datasets/ISIC 2016/unified_masks/"

######### train_img  (two 1742 img in isic_train_list)
save_dir = "C:/Datasets/ISIC_prepared/train/image/"
for i in isic_train_img_list:
    
    source_path = unified_img_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### train_mask
save_dir = "C:/Datasets/ISIC_prepared/train/mask/"
for i in isic_train_mask_list:
    
    source_path = unified_mask_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### Sanity check

img_dir = "C:/Datasets/ISIC_prepared/train/image/"
mask_dir = "C:/Datasets/ISIC_prepared/train/mask/" 

img_list = sorted(os.listdir(img_dir))
mask_list = sorted(os.listdir(mask_dir))

rand_num = np.random.randint(0, 808)

plt.figure(figsize=(15,15), facecolor="black", dpi=150)
plt.subplot(1,2,1)
img = img_dir + img_list[rand_num]
plt.imshow(plt.imread(img))
plt.title("Image", color="white")

plt.subplot(1,2,2)
msk = mask_dir + mask_list[rand_num]
plt.imshow(plt.imread(msk))
plt.title("Mask", color="white")

plt.show()

######### test_img  
save_dir = "C:/Datasets/ISIC_prepared/test/image/"
for i in isic_test_img_list:
    
    source_path = unified_img_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### train_mask
save_dir = "C:/Datasets/ISIC_prepared/test/mask/"
for i in isic_test_mask_list:
    
    source_path = unified_mask_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### Sanity check

img_dir = "C:/Datasets/ISIC_prepared/test/image/"
mask_dir = "C:/Datasets/ISIC_prepared/test/mask/" 

img_list = sorted(os.listdir(img_dir))
mask_list = sorted(os.listdir(mask_dir))

rand_num = np.random.randint(0, 378)

plt.figure(figsize=(15,15), facecolor="black", dpi=150)
plt.subplot(1,2,1)
img = img_dir + img_list[rand_num]
plt.imshow(plt.imread(img))
plt.title("Image", color="white")

plt.subplot(1,2,2)
msk = mask_dir + mask_list[rand_num]
plt.imshow(plt.imread(msk))
plt.title("Mask", color="white")

plt.show()


######### val_img  
save_dir = "C:/Datasets/ISIC_prepared/val/image/"
for i in isic_val_img_list:
    
    source_path = unified_img_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### val_mask
save_dir = "C:/Datasets/ISIC_prepared/val/mask/"
for i in isic_val_mask_list:
    
    source_path = unified_mask_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### Sanity check

img_dir = "C:/Datasets/ISIC_prepared/val/image/"
mask_dir = "C:/Datasets/ISIC_prepared/val/mask/" 

img_list = sorted(os.listdir(img_dir))
mask_list = sorted(os.listdir(mask_dir))

rand_num = np.random.randint(0, 89)

plt.figure(figsize=(15,15), facecolor="black", dpi=150)
plt.subplot(1,2,1)
img = img_dir + img_list[rand_num]
plt.imshow(plt.imread(img))
plt.title("Image", color="white")

plt.subplot(1,2,2)
msk = mask_dir + mask_list[rand_num]
plt.imshow(plt.imread(msk))
plt.title("Mask", color="white")

plt.show()


############################## kvasir  ##############################


kvasir_train = "F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/kvasir_polyp/anns/train.json"
kvasir_test = "F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/kvasir_polyp/anns/test.json"
kvasir_val = "F:/Visual Language Grounding (VLG)/Article Repo/MedSeg_EarlyFusion/text_data/kvasir_polyp/anns/val.json"

kvasir_train_json = pd.read_json(kvasir_train)
kvasir_test_json = pd.read_json(kvasir_test)
kvasir_val_json = pd.read_json(kvasir_val)

kvasir_train_img_list = list(kvasir_train_json.img_name)
kvasir_test_img_list = list(kvasir_test_json.img_name)
kvasir_val_img_list = list(kvasir_val_json.img_name)

kvasir_train_mask_list = list(kvasir_train_json.mask_name)
kvasir_test_mask_list = list(kvasir_test_json.mask_name)
kvasir_val_mask_list = list(kvasir_val_json.mask_name)


unified_kvasir_img_dir = "F:/Visual Language Grounding (VLG)/Datasets/Kvasir/unified_images/"
unified_kvasir_mask_dir = "F:/Visual Language Grounding (VLG)/Datasets/Kvasir/unified_masks/"

######### train_img  
save_dir = "C:/Datasets/Kvasir_prepared/train/image/"
for i in kvasir_train_img_list:
    
    source_path = unified_kvasir_img_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### train_mask
save_dir = "C:/Datasets/Kvasir_prepared/train/mask/"
for i in kvasir_train_mask_list:
    
    source_path = unified_kvasir_mask_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### Sanity check

img_dir = "C:/Datasets/Kvasir_prepared/train/image/"
mask_dir = "C:/Datasets/Kvasir_prepared/train/mask/" 

img_list = sorted(os.listdir(img_dir))
mask_list = sorted(os.listdir(mask_dir))

rand_num = np.random.randint(0, 89)

plt.figure(figsize=(15,15), facecolor="black", dpi=150)
plt.subplot(1,2,1)
img = img_dir + img_list[rand_num]
plt.imshow(plt.imread(img))
plt.title("Image", color="white")

plt.subplot(1,2,2)
msk = mask_dir + mask_list[rand_num]
plt.imshow(plt.imread(msk))
plt.title("Mask", color="white")

plt.show()


######### test_img  
save_dir = "C:/Datasets/Kvasir_prepared/test/image/"
for i in kvasir_test_img_list:
    
    source_path = unified_kvasir_img_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### test_mask
save_dir = "C:/Datasets/Kvasir_prepared/test/mask/"
for i in kvasir_test_mask_list:
    
    source_path = unified_kvasir_mask_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### Sanity check

img_dir = "C:/Datasets/Kvasir_prepared/test/image/"
mask_dir = "C:/Datasets/Kvasir_prepared/test/mask/" 

img_list = sorted(os.listdir(img_dir))
mask_list = sorted(os.listdir(mask_dir))

rand_num = np.random.randint(0, 89)

plt.figure(figsize=(15,15), facecolor="black", dpi=150)
plt.subplot(1,2,1)
img = img_dir + img_list[rand_num]
plt.imshow(plt.imread(img))
plt.title("Image", color="white")

plt.subplot(1,2,2)
msk = mask_dir + mask_list[rand_num]
plt.imshow(plt.imread(msk))
plt.title("Mask", color="white")

plt.show()

######### val_img  
save_dir = "C:/Datasets/Kvasir_prepared/val/image/"
for i in kvasir_val_img_list:
    
    source_path = unified_kvasir_img_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### val_mask
save_dir = "C:/Datasets/Kvasir_prepared/val/mask/"
for i in kvasir_val_mask_list:
    
    source_path = unified_kvasir_mask_dir + i
    
    target_path = save_dir + i
    
    shutil.copy(source_path, target_path)

######### Sanity check

img_dir = "C:/Datasets/Kvasir_prepared/val/image/"
mask_dir = "C:/Datasets/Kvasir_prepared/val/mask/" 

img_list = sorted(os.listdir(img_dir))
mask_list = sorted(os.listdir(mask_dir))

rand_num = np.random.randint(0, 89)

plt.figure(figsize=(15,15), facecolor="black", dpi=150)
plt.subplot(1,2,1)
img = img_dir + img_list[rand_num]
plt.imshow(plt.imread(img))
plt.title("Image", color="white")

plt.subplot(1,2,2)
msk = mask_dir + mask_list[rand_num]
plt.imshow(plt.imread(msk))
plt.title("Mask", color="white")

plt.show()


























