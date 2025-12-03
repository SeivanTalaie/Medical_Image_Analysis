######################################### libraries #########################################

import numpy as np 
import matplotlib.pyplot as plt 
import nibabel as nib 
from sklearn.preprocessing import MinMaxScaler 
from tifffile import imsave
import glob 
from keras.utils import to_categorical 
from tqdm import tqdm
from patchify import patchify

######################################### Numpy Creator #########################################

scaler=MinMaxScaler()

flair_list=sorted(glob.glob("F:\Python\Datasets\Brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*flair.nii"))
t1ce_list=sorted(glob.glob("F:\Python\Datasets\Brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*t1ce.nii"))
t2_list=sorted(glob.glob("F:\Python\Datasets\Brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*t2.nii"))
mask_list=sorted(glob.glob("F:\Python\Datasets\Brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData/*/*seg.nii"))

print(len(t1ce_list))

for img in tqdm(range(len(t1ce_list))):
    print("MRI images are being converted to numpy array")

    flair_train_img=nib.load(flair_list[img]).get_fdata()
    flair_train_img=scaler.fit_transform(flair_train_img.reshape(-1,flair_train_img.shape[-1])).reshape(flair_train_img.shape)
    flair_train_img=flair_train_img[56:184 , 56:184 , 13:141]
    flair_train_img=patchify(flair_train_img,(128,128,1) , step=1)
    flair_train_img=np.reshape(flair_train_img,(-1 , 128,128,1))
    
    t1ce_train_img=nib.load(t1ce_list[img]).get_fdata()
    t1ce_train_img=scaler.fit_transform(t1ce_train_img.reshape(-1,t1ce_train_img.shape[-1])).reshape(t1ce_train_img.shape)
    t1ce_train_img=t1ce_train_img[56:184 , 56:184 , 13:141]
    t1ce_train_img=patchify(t1ce_train_img,(128,128,1) , step=1)
    t1ce_train_img=np.reshape(t1ce_train_img,(-1 , 128,128,1))

    
    t2_train_img=nib.load(t2_list[img]).get_fdata()
    t2_train_img=scaler.fit_transform(t2_train_img.reshape(-1,t2_train_img.shape[-1])).reshape(t2_train_img.shape)
    t2_train_img=t2_train_img[56:184 , 56:184 , 13:141]
    t2_train_img=patchify(t2_train_img,(128,128,1) , step=1)
    t2_train_img=np.reshape(t2_train_img,(-1 , 128,128,1))
    
    training_mask=nib.load(mask_list[img]).get_fdata()
    training_mask=training_mask.astype("uint8") 
    training_mask[training_mask==4] = 3
    training_mask=training_mask[56:184 , 56:184 , 13:141]
    training_mask=patchify(training_mask,(128,128,1) , step=1)
    training_mask=np.reshape(training_mask,(-1 , 128,128,1))
    training_mask=to_categorical(training_mask , num_classes=4)
    
    combined=np.stack([flair_train_img , t1ce_train_img , t2_train_img ] ,axis=4)
    combined=np.reshape(combined, (128,128,128,3))
    
    np.save("F:/thesis project new/Numpy_arrays/image/image_" + str(img) + ".npy" , combined)
    np.save("F:/thesis project new/Numpy_arrays/mask/mask_" + str(img) + ".npy" , training_mask)



######################################### test the data #########################################

random_img=np.random.randint(0,368)

img=np.load("F:/thesis project new/Numpy_arrays/image/image_" +str(random_img) + ".npy")
msk=np.load("F:/thesis project new/Numpy_arrays/mask/mask_"   +str(random_img) + ".npy")

msk1=np.argmax(msk , axis=3)
n_slice=np.random.randint(0,img.shape[0]-1)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img[n_slice,:,:] , cmap="gray")
plt.title("image")
plt.subplot(1,2,2)
plt.imshow(msk1[n_slice,:,:] , cmap="gray")
plt.title("mask")
plt.show()

######################################### Split the data #########################################

import splitfolders 

input_folder = 'F:/thesis project new/Numpy_arrays'
output_folder = 'C:/Datasets/Brats2020/Input_data_2D'
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.80, .20), group_prefix=None)




















